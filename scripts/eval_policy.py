"""Roll out a trained ACT checkpoint and label every episode success or failure.

Same loop as `collect_demos.py`, with the scripted state machine replaced by the policy.
Measures success rate, and with --record writes the labelled rollouts the failure head
trains on.
"""

"""Launch Omniverse Toolkit first."""

import argparse
import atexit

from isaaclab.app import AppLauncher

# the published checkpoints. the v1 run is discarded, see phase 06
CHECKPOINTS = {
    "25k": "bjahoor/act-lift-cube-franka-v2-25k",
    "50k": "bjahoor/act-lift-cube-franka-v2",
}

# add argparse arguments
parser = argparse.ArgumentParser(description="Roll out a trained ACT policy on the lift task.")
parser.add_argument("--model", type=str, required=True, choices=list(CHECKPOINTS), help="Which trained checkpoint to roll out.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_rollouts", type=int, default=50, help="Stop after this many episodes.")
parser.add_argument("--record", action="store_true", default=False, help="Record a LeRobot dataset.")
parser.add_argument("--dataset_root", type=str, default="datasets/rollouts", help="Dataset directory.")
parser.add_argument("--repo_id", type=str, default="local/lift-cube-rollouts", help="LeRobot dataset repo id.")
parser.add_argument("--overwrite", action="store_true", default=False, help="Replace an existing dataset directory.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from lerobot_recorder import DROPPED, SUCCESS, TASK, TIMEOUT, EpisodeRecorder

# give up on an episode after this many steps. the expert finished in ~150, ACT is slower
MAX_EPISODE_STEPS = 500

# hold the cube at the goal this many steps before counting it a success
SUCCESS_STEPS = 25

# the open finger target the binary gripper command produced during collection, and the
# cutoff below which a predicted target counts as a close
FINGER_OPEN = 0.04
FINGER_CLOSE_BELOW = 0.03

# the arm joints. the fingers are left out of the state, matching collection
ARM_JOINTS = 7

# a single goal, matching collection
GOAL = mdp.UniformPoseCommandCfg.Ranges(
    pos_x=(0.5, 0.5), pos_y=(0.0, 0.0), pos_z=(0.375, 0.375),
    roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
)


def main():
    # parse configuration. the joint-command task, so the policy's 9 joint targets are
    # what the environment consumes. collection used IK-Abs, see phase 04
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # the demos were recorded under the stiffer controller that IK-Abs sets and this task does not
    env_cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # one term over every joint, so the action order is the articulation's own and matches
    # the recorded joint_pos_target. the stock task splits arm and a binary gripper, which does not
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False
    )
    env_cfg.actions.gripper_action = None

    # markers are rendered prims and would show up in the camera images
    env_cfg.commands.object_pose.debug_vis = False
    # the goal is resampled every 5 s by default, which is inside an episode here. the
    # policy cannot see the goal, so a moving one silently invalidates the success check
    env_cfg.commands.object_pose.resampling_time_range = (1.0e6, 1.0e6)
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
    env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
    # reset environment at start
    obs = env.reset()[0]

    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["object"]
    # the checkpoint predicts joint targets as 7 arm joints then the fingers
    joint_names = robot.data.joint_names
    assert joint_names[:7] == [f"panda_joint{i}" for i in range(1, 8)], joint_names
    assert all("finger" in n for n in joint_names[7:]), joint_names

    # one policy per env. ACT keeps a single action queue for the whole batch with no
    # per-env reset, so a shared instance would hand a freshly reset env another env's
    # chunk. identical weights, ~200 MB each
    checkpoint = CHECKPOINTS[args_cli.model]
    policies = []
    for _ in range(env.unwrapped.num_envs):
        policy = ACTPolicy.from_pretrained(checkpoint)
        policy.eval()
        policies.append(policy)
    # n_action_steps and chunk_size are baked into the checkpoint, do not override them
    preprocessor, postprocessor = make_pre_post_processors(policies[0].config, pretrained_path=checkpoint)

    # steps taken in the current episode, per env
    episode_steps = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
    # consecutive steps the cube has been at its goal, per env
    held_steps = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
    # the step after a reset pairs the old episode's images with the new episode's state
    skip_record = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)

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
            keep_failures=True,
        )
        atexit.register(recorder.close)
        # the full chunk, which TCE and ACM are computed from. lerobot's schema is fixed
        # per frame, so it goes beside the dataset rather than in it
        chunk_dir = Path(f"{args_cli.dataset_root}_chunks")
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[list[np.ndarray]] = [[] for _ in range(env.unwrapped.num_envs)]

    finished = 0
    succeeded_count = 0

    while simulation_app.is_running() and finished < args_cli.num_rollouts:
        # run everything in inference mode
        with torch.inference_mode():
            # the pair to record is the observation the action was computed from, so
            # capture it before stepping
            wrist = obs["policy"]["wrist_cam"]
            table = obs["policy"]["table_cam"]
            joint_pos = robot.data.joint_pos.clone()
            state = joint_pos[:, :ARM_JOINTS]

            # the env renders uint8 HWC, ACT wants float CHW normalized with the
            # checkpoint's own stats
            batch = {
                "observation.images.wrist": wrist.permute(0, 3, 1, 2).float() / 255.0,
                "observation.images.table": table.permute(0, 3, 1, 2).float() / 255.0,
                "observation.state": state,
                "task": [TASK] * env.unwrapped.num_envs,
            }
            batch = preprocessor(batch)
            # each env's own policy, one env's slice at a time, so the queues stay private
            actions = torch.zeros_like(joint_pos)
            chunk = [None] * env.unwrapped.num_envs
            for env_id, policy in enumerate(policies):
                one = {k: (v[env_id : env_id + 1] if torch.is_tensor(v) else v[:1]) for k, v in batch.items()}
                # select_action only predicts every n_action_steps, TCE needs one per step
                if recorder is not None:
                    chunk[env_id] = postprocessor(policy.predict_action_chunk(one))[0]
                actions[env_id] = postprocessor(policy.select_action(one))[0].to(env.unwrapped.device)
            # the demos only ever held the fingers fully open or fully closed, so ACT
            # regresses the midpoint when unsure and the gripper never grips. snap it back
            actions[:, 7:] = torch.where(actions[:, 7:] < FINGER_CLOSE_BELOW, 0.0, FINGER_OPEN)
            # the policy can ask for angles the joints do not have; physx clamps them anyway
            limits = robot.data.joint_pos_limits
            actions = actions.clamp(limits[..., 0], limits[..., 1])

            obs = env.step(actions)[0]
            episode_steps += 1

            # nothing resets on its own now, so decide here
            at_goal = success_term.func(env.unwrapped, **success_term.params)
            held_steps = torch.where(at_goal, held_steps + 1, torch.zeros_like(held_steps))
            succeeded = held_steps >= SUCCESS_STEPS
            dropped = dropped_term.func(env.unwrapped, **dropped_term.params)
            dones = succeeded | dropped | (episode_steps >= MAX_EPISODE_STEPS)

            if recorder is not None:
                object_pos = cube.data.root_pos_w - env.unwrapped.scene.env_origins
                for env_id in range(env.unwrapped.num_envs):
                    if skip_record[env_id]:
                        continue
                    recorder.add(
                        env_id, wrist[env_id], table[env_id], state[env_id], actions[env_id], object_pos[env_id]
                    )
                    # one chunk per recorded frame, or the two drift apart by a step
                    chunks[env_id].append(chunk[env_id].cpu().numpy())
                skip_record[:] = False

            # reset the finished envs and the policy's action queue
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                # reset returns fresh observations. without them ACT predicts its next
                # chunk from the finished episode's last frame and acts on it for 10 steps
                obs = env.unwrapped.reset(env_ids=done_ids)[0]
                # only the finished env's policy, the others are mid-chunk
                for env_id in done_ids.tolist():
                    policies[env_id].reset()
                episode_steps[done_ids] = 0
                held_steps[done_ids] = 0
                skip_record[done_ids] = True
                for env_id in done_ids.tolist():
                    finished += 1
                    succeeded_count += int(succeeded[env_id])
                    if recorder is not None:
                        reason = SUCCESS if succeeded[env_id] else (DROPPED if dropped[env_id] else TIMEOUT)
                        # named by the episode index the recorder is about to write
                        np.save(chunk_dir / f"episode_{recorder.saved:06d}.npy", np.stack(chunks[env_id]))
                        chunks[env_id] = []
                        recorder.finish(env_id, success=bool(succeeded[env_id]), reason=reason)
                print(f"[EVAL] {succeeded_count}/{finished} succeeded", flush=True)

    if recorder is not None:
        recorder.close()

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
