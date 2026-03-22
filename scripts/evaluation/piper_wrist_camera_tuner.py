"""Interactively tune the PiPER wrist camera offset without restarting Isaac Sim."""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import contextlib
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactively tune the PiPER wrist camera offset.")
parser.add_argument("--task", type=str, default="LeIsaac-PiPER-LiftCube-v0", help="Name of the task.")
parser.add_argument("--task_type", type=str, default="piperleader", help="Teleop/action task type.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=0.0, help="Episode length in seconds. 0 disables timeout.")
parser.add_argument("--zero_action_value", type=float, default=0.0, help="Constant action value to send every step.")
parser.add_argument("--pos_step", type=float, default=0.002, help="Position step size in meters.")
parser.add_argument("--rot_step_deg", type=float, default=2.0, help="Rotation step size in degrees.")
parser.add_argument(
    "--save_camera_dir",
    type=str,
    default="leisaac_outputs/piper_wrist_camera_tuner",
    help="Directory to save snapshots and pose logs.",
)
parser.add_argument(
    "--disable_camera_markers",
    action="store_true",
    help="Disable debug markers that visualize the base and wrist camera poses in the scene.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.sensors import Camera
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type
from PIL import Image

import leisaac  # noqa: F401


KEY_HELP = """
[Tuner] Key bindings
  Position (local offset):
    W/S: +X / -X
    A/D: +Y / -Y
    Q/E: +Z / -Z
  Rotation (local offset, degrees):
    U/O: +Roll / -Roll
    I/K: +Pitch / -Pitch
    J/L: +Yaw / -Yaw
  Other:
    P: save snapshot now
    R: reset environment
""".strip()


class RateLimiter:
    def __init__(self, hz: int):
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env: ManagerBasedRLEnv):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False
        self.save_state = False
        self.pos_delta = [0.0, 0.0, 0.0]
        self.rot_delta_deg = [0.0, 0.0, 0.0]
        self.change_labels: list[str] = []

    def __del__(self):
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def consume(self):
        state = {
            "reset": self.reset_state,
            "save": self.save_state,
            "pos_delta": self.pos_delta[:],
            "rot_delta_deg": self.rot_delta_deg[:],
            "labels": self.change_labels[:],
        }
        self.reset_state = False
        self.save_state = False
        self.pos_delta = [0.0, 0.0, 0.0]
        self.rot_delta_deg = [0.0, 0.0, 0.0]
        self.change_labels = []
        return state

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        name = event.input.name
        if name == "R":
            self.reset_state = True
            self.change_labels.append("reset")
        elif name == "P":
            self.save_state = True
            self.change_labels.append("save")
        elif name == "W":
            self.pos_delta[0] += args_cli.pos_step
            self.change_labels.append("pos_x+")
        elif name == "S":
            self.pos_delta[0] -= args_cli.pos_step
            self.change_labels.append("pos_x-")
        elif name == "A":
            self.pos_delta[1] += args_cli.pos_step
            self.change_labels.append("pos_y+")
        elif name == "D":
            self.pos_delta[1] -= args_cli.pos_step
            self.change_labels.append("pos_y-")
        elif name == "Q":
            self.pos_delta[2] += args_cli.pos_step
            self.change_labels.append("pos_z+")
        elif name == "E":
            self.pos_delta[2] -= args_cli.pos_step
            self.change_labels.append("pos_z-")
        elif name == "U":
            self.rot_delta_deg[0] += args_cli.rot_step_deg
            self.change_labels.append("roll+")
        elif name == "O":
            self.rot_delta_deg[0] -= args_cli.rot_step_deg
            self.change_labels.append("roll-")
        elif name == "I":
            self.rot_delta_deg[1] += args_cli.rot_step_deg
            self.change_labels.append("pitch+")
        elif name == "K":
            self.rot_delta_deg[1] -= args_cli.rot_step_deg
            self.change_labels.append("pitch-")
        elif name == "J":
            self.rot_delta_deg[2] += args_cli.rot_step_deg
            self.change_labels.append("yaw+")
        elif name == "L":
            self.rot_delta_deg[2] -= args_cli.rot_step_deg
            self.change_labels.append("yaw-")
        return True


def get_registered_camera_keys(env: ManagerBasedRLEnv) -> list[str]:
    return [key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)]


def save_camera_images(env: ManagerBasedRLEnv, output_dir: Path, camera_keys: list[str], tag: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    for camera_key in camera_keys:
        rgb = env.scene.sensors[camera_key].data.output["rgb"][0].detach().cpu().numpy()
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        image = Image.fromarray(rgb.astype("uint8"))
        image_path = output_dir / f"{tag}_{camera_key}.png"
        image.save(image_path)
        saved_paths[camera_key] = str(image_path)
    return saved_paths


def create_camera_markers():
    base_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/PiPERWristCameraTuner/BaseCameraMarker")
    base_cfg.markers["arrow"].scale = (0.12, 0.03, 0.03)
    wrist_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/PiPERWristCameraTuner/WristCameraMarker")
    wrist_cfg.markers["arrow"].scale = (0.12, 0.03, 0.03)
    return {
        "base": VisualizationMarkers(base_cfg),
        "wrist": VisualizationMarkers(wrist_cfg),
    }


def update_camera_markers(env: ManagerBasedRLEnv, camera_markers: dict[str, VisualizationMarkers]):
    for camera_key, marker in camera_markers.items():
        if camera_key not in env.scene.sensors:
            continue
        sensor = env.scene.sensors[camera_key]
        marker.visualize(sensor.data.pos_w, sensor.data.quat_w_world)


def apply_wrist_camera_pose(
    env: ManagerBasedRLEnv,
    wrist_body_idx: int,
    offset_pos: torch.Tensor,
    offset_rot_ros: torch.Tensor,
):
    robot = env.scene["robot"]
    wrist_camera: Camera = env.scene.sensors["wrist"]
    parent_pos = robot.data.body_pos_w[:, wrist_body_idx]
    parent_quat = robot.data.body_quat_w[:, wrist_body_idx]
    local_pos = offset_pos.unsqueeze(0).expand(env.num_envs, -1)
    local_rot_ros = offset_rot_ros.unsqueeze(0).expand(env.num_envs, -1)
    local_rot_world = math_utils.convert_camera_frame_orientation_convention(local_rot_ros, origin="ros", target="world")
    cam_pos_w, cam_quat_w = math_utils.combine_frame_transforms(parent_pos, parent_quat, local_pos, local_rot_world)
    wrist_camera.set_world_poses(cam_pos_w, cam_quat_w, convention="world")


def refresh_camera_outputs(env: ManagerBasedRLEnv, camera_keys: list[str]):
    env.sim.render()
    for camera_key in camera_keys:
        env.scene.sensors[camera_key].update(dt=0.0, force_recompute=True)


def apply_offset_delta(offset_pos: torch.Tensor, offset_rot_ros: torch.Tensor, control_state: dict):
    changed = False
    if any(abs(v) > 0.0 for v in control_state["pos_delta"]):
        offset_pos += torch.tensor(control_state["pos_delta"], device=offset_pos.device, dtype=offset_pos.dtype)
        changed = True
    if any(abs(v) > 0.0 for v in control_state["rot_delta_deg"]):
        delta_rad = torch.deg2rad(
            torch.tensor(control_state["rot_delta_deg"], device=offset_rot_ros.device, dtype=offset_rot_ros.dtype)
        )
        delta_quat = math_utils.quat_from_euler_xyz(
            delta_rad[0].unsqueeze(0), delta_rad[1].unsqueeze(0), delta_rad[2].unsqueeze(0)
        )[0]
        offset_rot_ros[:] = math_utils.quat_mul(offset_rot_ros.unsqueeze(0), delta_quat.unsqueeze(0))[0]
        changed = True
    return changed


def pose_summary(offset_pos: torch.Tensor, offset_rot_ros: torch.Tensor) -> dict:
    euler = torch.rad2deg(
        torch.stack(math_utils.euler_xyz_from_quat(offset_rot_ros.unsqueeze(0)), dim=-1).squeeze(0)
    )
    return {
        "offset_pos": [round(v, 6) for v in offset_pos.detach().cpu().tolist()],
        "offset_rot_ros_wxyz": [round(v, 6) for v in offset_rot_ros.detach().cpu().tolist()],
        "offset_rpy_deg_xyz": [round(v, 3) for v in euler.detach().cpu().tolist()],
    }


def save_snapshot(
    env: ManagerBasedRLEnv,
    output_dir: Path,
    camera_keys: list[str],
    snapshot_index: int,
    offset_pos: torch.Tensor,
    offset_rot_ros: torch.Tensor,
    labels: list[str],
):
    tag = f"snapshot_{snapshot_index:04d}"
    image_paths = save_camera_images(env, output_dir, camera_keys, tag=tag)
    summary = pose_summary(offset_pos, offset_rot_ros)
    payload = {
        "snapshot": snapshot_index,
        "labels": labels,
        "timestamp": time.time(),
        **summary,
        "image_paths": image_paths,
    }
    json_path = output_dir / f"{tag}_wrist_offset.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with (output_dir / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    print(f"[Tuner] saved snapshot {snapshot_index} -> {json_path}")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task, task_type=args_cli.task_type)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.recorders = None
    env_cfg.scene.wrist.update_period = 0.0
    env_cfg.scene.base.update_period = 0.0

    if args_cli.episode_length_s <= 0.0:
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    else:
        env_cfg.episode_length_s = args_cli.episode_length_s

    initial_offset_pos = torch.tensor(env_cfg.scene.wrist.offset.pos, dtype=torch.float32, device=args_cli.device)
    initial_offset_rot_ros = torch.tensor(env_cfg.scene.wrist.offset.rot, dtype=torch.float32, device=args_cli.device)

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    robot = env.scene["robot"]
    wrist_body_idx = robot.find_bodies("link6")[0][0]
    offset_pos = initial_offset_pos.clone()
    offset_rot_ros = initial_offset_rot_ros.clone()

    controller = Controller()
    rate_limiter = RateLimiter(args_cli.step_hz)
    action_dim = env.action_manager.total_action_dim
    zero_action = torch.full((env.num_envs, action_dim), args_cli.zero_action_value, device=env.device)
    camera_keys = get_registered_camera_keys(env)
    output_dir = Path(args_cli.save_camera_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_markers = None if args_cli.disable_camera_markers else create_camera_markers()

    obs_dict, _ = env.reset()
    apply_wrist_camera_pose(env, wrist_body_idx, offset_pos, offset_rot_ros)
    refresh_camera_outputs(env, camera_keys)

    print(f"[Tuner] task={args_cli.task}, task_type={task_type}, action_dim={action_dim}")
    print(f"[Tuner] registered cameras={camera_keys}")
    print(f"[Tuner] save dir={output_dir}")
    if camera_markers is not None:
        print("[Tuner] camera markers enabled: base=red arrow, wrist=blue arrow")
    print(KEY_HELP)
    print(f"[Tuner] initial pose={pose_summary(offset_pos, offset_rot_ros)}")
    if "policy" in obs_dict and "joint_pos" in obs_dict["policy"]:
        print(f"[Tuner] reset joint_pos={obs_dict['policy']['joint_pos'][0].tolist()}")

    snapshot_index = 0
    pending_snapshot_labels = ["startup"]

    with contextlib.suppress(KeyboardInterrupt):
        with torch.inference_mode():
            while simulation_app.is_running() and not simulation_app.is_exiting():
                control_state = controller.consume()
                changed = apply_offset_delta(offset_pos, offset_rot_ros, control_state)

                if changed:
                    summary = pose_summary(offset_pos, offset_rot_ros)
                    print(f"[Tuner] updated pose via {control_state['labels']}: {summary}")
                    pending_snapshot_labels = control_state["labels"][:]

                if control_state["reset"]:
                    obs_dict, _ = env.reset()
                    if "policy" in obs_dict and "joint_pos" in obs_dict["policy"]:
                        print(f"[Tuner] reset joint_pos={obs_dict['policy']['joint_pos'][0].tolist()}")

                apply_wrist_camera_pose(env, wrist_body_idx, offset_pos, offset_rot_ros)

                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, task_type)

                env.step(zero_action)
                apply_wrist_camera_pose(env, wrist_body_idx, offset_pos, offset_rot_ros)
                refresh_camera_outputs(env, camera_keys)

                if camera_markers is not None:
                    update_camera_markers(env, camera_markers)

                if changed or control_state["save"] or snapshot_index == 0:
                    save_snapshot(
                        env,
                        output_dir,
                        camera_keys,
                        snapshot_index,
                        offset_pos,
                        offset_rot_ros,
                        labels=pending_snapshot_labels if pending_snapshot_labels else control_state["labels"],
                    )
                    snapshot_index += 1
                    pending_snapshot_labels = []

                rate_limiter.sleep(env)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
