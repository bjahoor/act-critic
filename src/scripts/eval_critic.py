"""Roll out ACT with the critic head attached and show the failure score live.

The demo script. Same environment and loop as `eval_policy.py`, with the head in the loop
and an on-screen readout, so a person watching the stream sees the score move while the
robot works. Nothing is recorded — `eval_policy.py` owns that.

    LIVESTREAM=1 PUBLIC_IP=<server-ip> PYTHONEXE=$PWD/.venv-lerobot/bin/python \
      ~/isaacsim/python.sh src/scripts/eval_critic.py --model 20k --num_envs 4 --enable_cameras
"""

"""Launch Omniverse Toolkit first."""

import sys
from pathlib import Path

# src/ on the path, so `recording` and `modeling_act_critic` import however this is launched
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from isaaclab.app import AppLauncher

# the published checkpoints. the v1 run is discarded, see phase 06
CHECKPOINTS = {
    "10k": "bjahoor/act-lift-cube-franka-v2-10k",
    "20k": "bjahoor/act-lift-cube-franka-v2-20k",
    "30k": "bjahoor/act-lift-cube-franka-v2-30k",
    "40k": "bjahoor/act-lift-cube-franka-v2-40k",
    "50k": "bjahoor/act-lift-cube-franka-v2",
}

parser = argparse.ArgumentParser(description="Roll out ACT with the critic head and display the score.")
parser.add_argument("--model", type=str, default="20k", choices=list(CHECKPOINTS), help="Which trained checkpoint to roll out.")
parser.add_argument("--critic", type=str, default="checkpoints/critic-abmil/critic.pt", help="Trained head. Untrained if absent.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--num_rollouts", type=int, default=50, help="Stop after this many episodes.")
parser.add_argument("--threshold", type=float, default=0.5, help="Score above which the readout reads FAILING.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections import deque

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

from huggingface_hub import snapshot_download
from lerobot.policies.factory import make_pre_post_processors
from safetensors.torch import load_file

from modeling_act_critic import HISTORY_OFFSETS, ACTWithCritic
from recording.lerobot_recorder import TASK

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

# must match train_critic.TCE_GAP, or the head sees a statistic it was not trained on
TCE_GAP = 10

# a single goal, matching collection
GOAL = mdp.UniformPoseCommandCfg.Ranges(
    pos_x=(0.5, 0.5), pos_y=(0.0, 0.0), pos_z=(0.375, 0.375),
    roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
)


class ScorePanel:
    """A Kit window showing the live failure score, one bar and one plot line per env.

    `omni.ui` is Isaac Sim's own toolkit and the stream carries the whole application
    window, so this reaches a remote viewer. Drawing into the scene instead would put the
    readout inside `wrist_cam` and `table_cam`, which are the policy's observations.
    """

    def __init__(self, num_envs: int, threshold: float):
        import omni.ui as ui
        from isaaclab.ui.widgets import LiveLinePlot

        self.threshold = threshold
        self.bars, self.labels = [], []
        self.window = ui.Window(
            "Failure Score", width=340, height=120 + 34 * num_envs,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )
        with self.window.frame:
            with ui.VStack(spacing=4, height=0):
                for env_id in range(num_envs):
                    with ui.HStack(height=24, spacing=6):
                        ui.Label(f"env {env_id}", width=48)
                        model = ui.SimpleFloatModel(0.0)
                        ui.ProgressBar(model=model)
                        self.bars.append(model)
                        self.labels.append(ui.Label("--", width=76))
                # built inside the parented frame: its own __init__ reads attributes that
                # are only assigned afterwards, and survives that solely because
                # ui.Frame(build_fn=...) builds lazily
                self.plot = LiveLinePlot(
                    y_data=[[] for _ in range(num_envs)],
                    y_min=0.0, y_max=1.0, plot_height=140,
                    legends=[f"env {i}" for i in range(num_envs)],
                    max_datapoints=200,
                )

    def update(self, scores: list[float | None]) -> None:
        for model, label, score in zip(self.bars, self.labels, scores):
            if score is None:
                model.set_value(0.0)
                label.text = "warmup"
                continue
            model.set_value(score)
            label.text = f"{score:.2f} {'FAILING' if score >= self.threshold else 'ok'}"
        self.plot.add_datapoint([0.0 if s is None else s for s in scores])


def tce_acm(history: deque, mean: np.ndarray, std: np.ndarray, scale: dict) -> tuple[float, float]:
    """The current chunk's TCE and ACM, standardized exactly as training did.

    Mirrors `train_critic.compute_tce_acm` for a single frame. A drift between the two
    would feed the head a statistic on a scale it never saw, with no visible error.
    """
    c = (history[-1] - mean) / std
    acm = float(np.sqrt((c**2).mean()))
    # the chunk from TCE_GAP steps ago overlaps this one by chunk_size - TCE_GAP
    tce = 0.0
    if len(history) > TCE_GAP:
        past = (history[-1 - TCE_GAP] - mean) / std
        tce = float(((c[:-TCE_GAP] - past[TCE_GAP:]) ** 2).mean())
    return (
        (tce - scale["tce_mean"]) / scale["tce_std"],
        (acm - scale["acm_mean"]) / scale["acm_std"],
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

    env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
    obs = env.reset()[0]

    robot = env.unwrapped.scene["robot"]
    # the checkpoint predicts joint targets as 7 arm joints then the fingers
    joint_names = robot.data.joint_names
    assert joint_names[:7] == [f"panda_joint{i}" for i in range(1, 8)], joint_names
    assert all("finger" in n for n in joint_names[7:]), joint_names

    checkpoint = CHECKPOINTS[args_cli.model]

    # the head and the scaler its TCE and ACM were standardized with, saved together by
    # train_critic. absent, the head is random and the readout is noise, which is still
    # enough to check the display itself
    critic_path = Path(args_cli.critic)
    trained = critic_path.is_file()
    if trained:
        saved = torch.load(critic_path, map_location="cpu", weights_only=False)
        scale, pooling = saved["scale"], saved["pooling"]
        if saved["base"] != checkpoint:
            print(f"[WARN] head was trained on {saved['base']}, rolling out {checkpoint}", flush=True)
    else:
        print(f"[WARN] {critic_path} not found, the head is untrained and the score is noise", flush=True)
        scale = {"tce_mean": 0.0, "tce_std": 1.0, "acm_mean": 0.0, "acm_std": 1.0}
        pooling = "abmil"

    # one policy per env. ACT keeps a single action queue for the whole batch with no
    # per-env reset, so a shared instance would hand a freshly reset env another env's
    # chunk. the head keeps its own frame history per env for the same reason
    policies = []
    for _ in range(env.unwrapped.num_envs):
        policy = ACTWithCritic.from_pretrained(checkpoint, pooling=pooling)
        if trained:
            policy.critic.load_state_dict(saved["head"])
        policy.eval()
        policies.append(policy)
    # n_action_steps and chunk_size are baked into the checkpoint, do not override them
    preprocessor, postprocessor = make_pre_post_processors(policies[0].config, pretrained_path=checkpoint)

    # the action statistics the chunks are normalized by before TCE and ACM, the same ones
    # training used. raw, the arm's radians dwarf the fingers' metres
    norm = load_file(Path(snapshot_download(checkpoint)) / "policy_preprocessor_step_3_normalizer_processor.safetensors")
    act_mean, act_std = norm["action.mean"].numpy(), norm["action.std"].numpy()

    # the last TCE_GAP+1 chunks per env, which TCE is computed across
    chunk_history = [deque(maxlen=TCE_GAP + 1) for _ in range(env.unwrapped.num_envs)]

    # only build the panel when something can display it, so the same script still runs
    # under a plain headless launch
    panel = ScorePanel(env.unwrapped.num_envs, args_cli.threshold) if env.unwrapped.sim.has_gui() else None

    episode_steps = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
    held_steps = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)

    finished = 0
    succeeded_count = 0

    while simulation_app.is_running() and finished < args_cli.num_rollouts:
        with torch.inference_mode():
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

            actions = torch.zeros_like(joint_pos)
            scores: list[float | None] = [None] * env.unwrapped.num_envs
            for env_id, policy in enumerate(policies):
                # some pipeline entries are None, and task is a plain list
                one = {
                    k: v[env_id : env_id + 1] if torch.is_tensor(v) else (v[:1] if isinstance(v, list) else v)
                    for k, v in batch.items()
                }
                # TCE and ACM come from this frame's chunk, so the chunk is needed before
                # the head can be called with them
                chunk = postprocessor(policy.predict_action_chunk(one))[0]
                chunk_history[env_id].append(chunk.cpu().numpy())
                tce, acm = tce_acm(chunk_history[env_id], act_mean, act_std, scale)
                device = env.unwrapped.device
                out = policy.critic_score(
                    one,
                    torch.tensor([[tce]], dtype=torch.float32, device=device),
                    torch.tensor([[acm]], dtype=torch.float32, device=device),
                )
                # None until the head's four-frame history fills, 0.3 s in
                if out is not None:
                    scores[env_id] = float(out["failure_score"])
                actions[env_id] = postprocessor(policy.select_action(one))[0].to(device)
            # the demos only ever held the fingers fully open or fully closed, so ACT
            # regresses the midpoint when unsure and the gripper never grips. snap it back
            actions[:, 7:] = torch.where(actions[:, 7:] < FINGER_CLOSE_BELOW, 0.0, FINGER_OPEN)
            # the policy can ask for angles the joints do not have; physx clamps them anyway
            limits = robot.data.joint_pos_limits
            actions = actions.clamp(limits[..., 0], limits[..., 1])

            obs = env.step(actions)[0]
            episode_steps += 1

            if panel is not None:
                panel.update(scores)

            # nothing resets on its own now, so decide here
            at_goal = success_term.func(env.unwrapped, **success_term.params)
            held_steps = torch.where(at_goal, held_steps + 1, torch.zeros_like(held_steps))
            succeeded = held_steps >= SUCCESS_STEPS
            dropped = dropped_term.func(env.unwrapped, **dropped_term.params)
            dones = succeeded | dropped | (episode_steps >= MAX_EPISODE_STEPS)

            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                # reset returns fresh observations. without them ACT predicts its next
                # chunk from the finished episode's last frame and acts on it for 10 steps
                obs = env.unwrapped.reset(env_ids=done_ids)[0]
                # only the finished env's policy, the others are mid-chunk. this clears the
                # head's frame history too
                for env_id in done_ids.tolist():
                    policies[env_id].reset()
                    chunk_history[env_id].clear()
                episode_steps[done_ids] = 0
                held_steps[done_ids] = 0
                for env_id in done_ids.tolist():
                    finished += 1
                    succeeded_count += int(succeeded[env_id])
                print(f"[EVAL] {succeeded_count}/{finished} succeeded", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
