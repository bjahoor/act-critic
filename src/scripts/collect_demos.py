# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Collect scripted pick-and-lift demonstrations as a LeRobot dataset.

Adapted from Isaac Lab's scripts/environments/state_machine/lift_cube_sm.py. The state machine is
implemented in the warp kernel `infer_state_machine` and runs on the GPU across all environments.

.. code-block:: bash

    PYTHONEXE=$PWD/.venv-lerobot/bin/python PYTHONPATH=$PWD/src ~/isaacsim/python.sh \
      src/scripts/collect_demos.py \
      --num_envs 10 --enable_cameras --record --num_demos 100 --overwrite

"""

"""Launch Omniverse Toolkit first."""

import sys
from pathlib import Path

# src/ on the path, so `recording` imports however this is launched
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import atexit

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--record", action="store_true", default=False, help="Record a LeRobot dataset.")
parser.add_argument("--num_demos", type=int, default=50, help="Stop after this many successful episodes.")
parser.add_argument("--dataset_root", type=str, default="datasets/lift-cube-franka", help="Dataset directory.")
parser.add_argument("--repo_id", type=str, default="local/lift-cube-franka", help="LeRobot dataset repo id.")
parser.add_argument("--overwrite", action="store_true", default=False, help="Replace an existing dataset directory.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from recording.lerobot_recorder import EpisodeRecorder

# give up on an episode after this many steps
MAX_EPISODE_STEPS = 500

# hold the cube at the goal this many steps before counting it a success
SUCCESS_STEPS = 25

# the arm joints. the finger positions are deliberately left out of the recorded state:
# with them in, the policy learns to copy its own gripper state and never closes
ARM_JOINTS = 7

# a single goal, the centre of the range the task ships
GOAL = mdp.UniformPoseCommandCfg.Ranges(
    pos_x=(0.5, 0.5), pos_y=(0.0, 0.0), pos_z=(0.375, 0.375),
    roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
)

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # markers are rendered prims and would show up in the camera images
    env_cfg.commands.object_pose.debug_vis = not args_cli.record
    # the goal is randomized per episode but the policy never sees it, so the task is not
    # learnable from pixels. one fixed point instead
    env_cfg.commands.object_pose.ranges = GOAL

    # the lift task ships this check but does not register it. keep it to call ourselves,
    # and stop the clock so nothing resets behind us mid-episode
    success_term = DoneTerm(func=lift_mdp.object_reached_goal, params={"threshold": 0.1})
    env_cfg.terminations.time_out = None
    # a dropped cube would otherwise reset the env inside env.step(), behind our back
    dropped_term = env_cfg.terminations.object_dropping
    env_cfg.terminations.object_dropping = None

    # add cameras to the scene
    env_cfg.scene.wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=200,
        width=200,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.3), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        ),
    )
    env_cfg.scene.table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=200,
        width=200,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.4, 0.0, 0.7), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
        ),
    )

    # expose the camera images as observations
    env_cfg.observations.policy.wrist_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
    )
    env_cfg.observations.policy.table_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
    )
    # images cannot be concatenated with the state vector
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    obs = env.reset()[0]

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01
    )
    # steps taken in the current episode, per env
    episode_steps = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
    # consecutive steps the cube has been at its goal, per env
    held_steps = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
    # the step after a reset pairs the old episode's images with the new episode's state
    skip_record = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    # recorded action is joint_pos_target, which must line up as 7 arm joints then the fingers
    joint_names = robot.data.joint_names
    assert joint_names[:7] == [f"panda_joint{i}" for i in range(1, 8)], joint_names
    assert all("finger" in n for n in joint_names[7:]), joint_names
    recorder = None
    if args_cli.record:
        recorder = EpisodeRecorder(
            repo_id=args_cli.repo_id,
            root=args_cli.dataset_root,
            num_envs=env.unwrapped.num_envs,
            state_dim=ARM_JOINTS,
            action_dim=robot.data.joint_pos.shape[1],
            image_shape=(env_cfg.scene.wrist_cam.height, env_cfg.scene.wrist_cam.width, 3),
            fps=round(1.0 / (env_cfg.sim.dt * env_cfg.decimation)),
            overwrite=args_cli.overwrite,
        )
        atexit.register(recorder.close)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            # the pair to record is the observation the action was computed from, so
            # capture it before stepping. the joint target only exists after.
            prev_obs = obs
            prev_joint_pos = robot.data.joint_pos.clone()

            obs = env.step(actions)[0]
            episode_steps += 1

            # nothing resets on its own now, so decide here
            at_goal = success_term.func(env.unwrapped, **success_term.params)
            held_steps = torch.where(at_goal, held_steps + 1, torch.zeros_like(held_steps))
            succeeded = held_steps >= SUCCESS_STEPS
            dropped = dropped_term.func(env.unwrapped, **dropped_term.params)
            dones = succeeded | dropped | (episode_steps >= MAX_EPISODE_STEPS)

            if recorder is not None:
                wrist = prev_obs["policy"]["wrist_cam"]
                table = prev_obs["policy"]["table_cam"]
                joint_pos = prev_joint_pos[:, :ARM_JOINTS]
                # the IK can ask for angles the joints do not have; physx clamps them when
                # moving the arm, so clamp the labels too rather than teaching impossible commands
                limits = robot.data.joint_pos_limits
                joint_target = robot.data.joint_pos_target.clamp(limits[..., 0], limits[..., 1])
                for env_id in range(env.unwrapped.num_envs):
                    if skip_record[env_id]:
                        continue
                    recorder.add(env_id, wrist[env_id], table[env_id], joint_pos[env_id], joint_target[env_id])
                skip_record[:] = False

            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            # reset the finished envs and their state machines
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                env.unwrapped.reset(env_ids=done_ids)
                pick_sm.reset_idx(done_ids)
                episode_steps[done_ids] = 0
                held_steps[done_ids] = 0
                skip_record[done_ids] = True
                if recorder is not None:
                    for env_id in done_ids.tolist():
                        recorder.finish(env_id, success=bool(succeeded[env_id]))
                    if recorder.saved >= args_cli.num_demos:
                        print(f"[RECORD] saved {recorder.saved} episodes", flush=True)
                        break

    if recorder is not None:
        recorder.close()

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
