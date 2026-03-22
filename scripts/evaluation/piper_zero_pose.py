"""Visualize PiPER zero pose without any policy or VLA control."""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import contextlib
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize PiPER zero pose without policy control.")
parser.add_argument("--task", type=str, default="LeIsaac-PiPER-LiftCube-v0", help="Name of the task.")
parser.add_argument("--task_type", type=str, default="piperleader", help="Teleop/action task type.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=0.0, help="Episode length in seconds. 0 disables timeout.")
parser.add_argument("--zero_action_value", type=float, default=0.0, help="Constant action value to send every step.")
parser.add_argument(
    "--save_camera_dir",
    type=str,
    default="leisaac_outputs/piper_zero_pose",
    help="Directory to save registered camera RGB images as PNG files.",
)
parser.add_argument(
    "--disable_camera_dump",
    action="store_true",
    help="Disable automatic PNG dumps for the registered cameras.",
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
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type
from PIL import Image

import leisaac  # noqa: F401


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env: ManagerBasedRLEnv):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
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

    def __del__(self):
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "R":
            self.reset_state = True
        return True


def get_registered_camera_keys(env: ManagerBasedRLEnv) -> list[str]:
    return [key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)]


def save_camera_images(env: ManagerBasedRLEnv, output_dir: Path, camera_keys: list[str], tag: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera_key in camera_keys:
        rgb = env.scene.sensors[camera_key].data.output["rgb"][0].detach().cpu().numpy()
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        image = Image.fromarray(rgb.astype("uint8"))
        image_path = output_dir / f"{tag}_{camera_key}.png"
        image.save(image_path)
        print(f"[ZeroPose] saved {camera_key} image to {image_path}")


def create_camera_markers():
    base_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/PiPERZeroPose/BaseCameraMarker")
    base_cfg.markers["arrow"].scale = (0.12, 0.03, 0.03)
    wrist_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/PiPERZeroPose/WristCameraMarker")
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


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task, task_type=args_cli.task_type)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.recorders = None

    if args_cli.episode_length_s <= 0.0:
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    else:
        env_cfg.episode_length_s = args_cli.episode_length_s

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    controller = Controller()
    rate_limiter = RateLimiter(args_cli.step_hz)
    action_dim = env.action_manager.total_action_dim
    zero_action = torch.full((env.num_envs, action_dim), args_cli.zero_action_value, device=env.device)
    camera_keys = get_registered_camera_keys(env)
    output_dir = Path(args_cli.save_camera_dir)
    camera_markers = None if args_cli.disable_camera_markers else create_camera_markers()

    obs_dict, _ = env.reset()
    controller.reset()

    print(
        f"[ZeroPose] task={args_cli.task}, task_type={task_type}, action_dim={action_dim},"
        f" zero_action_value={args_cli.zero_action_value}"
    )
    print(f"[ZeroPose] registered cameras={camera_keys}")
    if camera_markers is not None:
        print("[ZeroPose] camera markers enabled: base=red arrow, wrist=blue arrow")
    if "policy" in obs_dict and "joint_pos" in obs_dict["policy"]:
        print(f"[ZeroPose] reset joint_pos={obs_dict['policy']['joint_pos'][0].tolist()}")

    for _ in range(5):
        env.sim.render()
    if camera_markers is not None:
        update_camera_markers(env, camera_markers)
    if not args_cli.disable_camera_dump:
        save_camera_images(env, output_dir, camera_keys, tag="reset_000")

    reset_count = 0
    with contextlib.suppress(KeyboardInterrupt):
        with torch.inference_mode():
            while simulation_app.is_running() and not simulation_app.is_exiting():
                if controller.reset_state:
                    obs_dict, _ = env.reset()
                    controller.reset()
                    reset_count += 1
                    for _ in range(5):
                        env.sim.render()
                    if camera_markers is not None:
                        update_camera_markers(env, camera_markers)
                    if "policy" in obs_dict and "joint_pos" in obs_dict["policy"]:
                        print(f"[ZeroPose] reset joint_pos={obs_dict['policy']['joint_pos'][0].tolist()}")
                    if not args_cli.disable_camera_dump:
                        save_camera_images(env, output_dir, camera_keys, tag=f"reset_{reset_count:03d}")

                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, task_type)

                env.step(zero_action)
                if camera_markers is not None:
                    update_camera_markers(env, camera_markers)
                rate_limiter.sleep(env)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
