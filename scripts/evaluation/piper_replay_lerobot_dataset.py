"""Replay a converted LeRobot PiPER IsaacLab HDF5 dataset in simulation."""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import copy
import contextlib
import os
import time
from math import ceil, sqrt
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay converted LeRobot PiPER dataset in IsaacLab.")
parser.add_argument("--task", type=str, default="LeIsaac-PiPER-LiftCube-v0", help="Name of the task.")
parser.add_argument("--task_type", type=str, default="piperleader", help="Teleop/action task type.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=0.0, help="Episode length in seconds. 0 disables timeout.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="leisaac_outputs/piper_lerobot_replay.hdf5",
    help="Converted IsaacLab-compatible replay HDF5 path.",
)
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="Episode indices to replay. Empty means replay all episodes in the file.",
)
parser.add_argument(
    "--override_initial_robot_state",
    action="store_true",
    help="Ignore converted initial robot joints and keep the task's default reset robot state.",
)
parser.add_argument(
    "--save_replay_video_dir",
    type=str,
    default="leisaac_outputs/piper_replay_videos",
    help="Directory to save per-replayed-episode camera videos.",
)
parser.add_argument(
    "--disable_replay_video",
    action="store_true",
    help="Disable per-replayed-episode camera video dumps.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import Camera
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type

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


def get_registered_camera_keys(env: ManagerBasedRLEnv) -> list[str]:
    return [key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)]


def to_uint8_rgb_frame(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    while frame.ndim > 3 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"Expected a camera frame with shape (H, W, C), got {frame.shape}")
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


class EpisodeVideoRecorder:
    def __init__(self, output_dir: Path, camera_keys: list[str], fps: int, enabled: bool = True):
        self.output_dir = output_dir
        self.camera_keys = camera_keys
        self.fps = fps
        self.enabled = enabled and len(camera_keys) > 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._cv2 = None
        self._writer = None
        self._current_episode = None
        self._pending_path = None
        self._frame_count = 0
        self._last_frame = None
        self._last_timestamp = None

    def start_episode(self, episode_name: str):
        if not self.enabled:
            return
        self.close(status="interrupted")
        safe_episode_name = episode_name.replace("/", "_")
        self._current_episode = safe_episode_name
        self._pending_path = self.output_dir / f"{safe_episode_name}_pending.mp4"
        self._frame_count = 0
        self._last_frame = None
        self._last_timestamp = None

    def add_observation(self, observation_dict: dict, timestamp: float | None = None):
        if not self.enabled or self._current_episode is None:
            return
        if timestamp is None:
            timestamp = time.time()
        camera_frames = {
            camera_key: observation_dict[camera_key]
            for camera_key in self.camera_keys
            if camera_key in observation_dict
        }
        if len(camera_frames) == 0:
            return

        frame = self._compose_frame(camera_frames)
        if self._writer is None:
            self._open_writer(width=frame.shape[1], height=frame.shape[0])
        if self._last_frame is None:
            self._last_frame = frame
            self._last_timestamp = timestamp
            return

        elapsed = max(0.0, timestamp - self._last_timestamp)
        repeat_count = max(1, int(round(elapsed * self.fps)))
        self._write_frame(self._last_frame, repeat_count=repeat_count)
        self._last_frame = frame
        self._last_timestamp = timestamp

    def close(self, status: str):
        if not self.enabled or self._current_episode is None:
            return None

        if self._last_frame is not None:
            self._write_frame(self._last_frame, repeat_count=1)
            self._last_frame = None
            self._last_timestamp = None

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        final_path = None
        if self._pending_path is not None and self._pending_path.exists():
            final_path = self.output_dir / f"{self._current_episode}_{status}.mp4"
            self._pending_path.replace(final_path)
            print(
                f"[Replay] Saved episode video to {final_path}"
                f" ({self._frame_count} frames, status={status})."
            )
        elif self._frame_count == 0:
            print(f"[Replay] Skipped video for episode {self._current_episode}: no camera frames were available.")

        self._current_episode = None
        self._pending_path = None
        self._frame_count = 0
        return final_path

    def _ensure_cv2(self):
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2

    def _open_writer(self, width: int, height: int):
        self._ensure_cv2()
        if self._pending_path is None:
            raise RuntimeError("Video recorder has not been started for the current episode.")
        fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = self._cv2.VideoWriter(str(self._pending_path), fourcc, float(self.fps), (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self._pending_path}")

    def _write_frame(self, frame: np.ndarray, repeat_count: int):
        if self._writer is None:
            raise RuntimeError("Video writer must be opened before writing frames.")
        bgr_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
        for _ in range(repeat_count):
            self._writer.write(bgr_frame)
        self._frame_count += repeat_count

    def _compose_frame(self, camera_frames: dict[str, np.ndarray]) -> np.ndarray:
        ordered_frames = []
        for camera_key in self.camera_keys:
            if camera_key in camera_frames:
                ordered_frames.append((camera_key, to_uint8_rgb_frame(camera_frames[camera_key])))
        for camera_key, frame in camera_frames.items():
            if camera_key not in self.camera_keys:
                ordered_frames.append((camera_key, to_uint8_rgb_frame(frame)))
        if len(ordered_frames) == 0:
            raise ValueError("No camera frames available to compose into a video frame.")

        self._ensure_cv2()

        cell_height = max(frame.shape[0] for _, frame in ordered_frames)
        cell_width = max(frame.shape[1] for _, frame in ordered_frames)
        title_height = 28
        column_count = 1 if len(ordered_frames) == 1 else int(ceil(sqrt(len(ordered_frames))))
        row_count = int(ceil(len(ordered_frames) / column_count))
        canvas = np.zeros((row_count * (cell_height + title_height), column_count * cell_width, 3), dtype=np.uint8)

        for index, (camera_key, frame) in enumerate(ordered_frames):
            row = index // column_count
            col = index % column_count
            x0 = col * cell_width
            y0 = row * (cell_height + title_height)
            frame_height, frame_width = frame.shape[:2]
            canvas[y0 + title_height : y0 + title_height + frame_height, x0 : x0 + frame_width] = frame
            self._cv2.putText(
                canvas,
                camera_key,
                (x0 + 8, y0 + 19),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                self._cv2.LINE_AA,
            )

        return canvas


def sort_episode_names(episode_names: list[str]) -> list[str]:
    def sort_key(name: str):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name

    return sorted(episode_names, key=sort_key)


def build_episode_reset_state(
    default_reset_state: dict,
    episode_initial_state: dict | None,
    override_initial_robot_state: bool,
):
    reset_state = copy.deepcopy(default_reset_state)
    if override_initial_robot_state or episode_initial_state is None:
        return reset_state

    robot_state = episode_initial_state.get("articulation", {}).get("robot")
    if robot_state is None:
        raise KeyError("Episode initial_state is missing articulation.robot")

    reset_state["articulation"]["robot"]["joint_position"] = robot_state["joint_position"].clone()
    reset_state["articulation"]["robot"]["joint_velocity"] = robot_state["joint_velocity"].clone()
    return reset_state


def main():
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    episode_names = sort_episode_names(list(dataset_file_handler.get_episode_names()))
    episode_count = len(episode_names)
    if episode_count == 0:
        raise ValueError("No episodes found in the converted replay dataset.")

    episode_indices = args_cli.select_episodes or list(range(episode_count))
    for episode_index in episode_indices:
        if episode_index < 0 or episode_index >= episode_count:
            raise IndexError(f"Requested episode {episode_index} is outside valid range [0, {episode_count - 1}].")

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
    action_dim = env.action_manager.total_action_dim
    rate_limiter = RateLimiter(args_cli.step_hz)
    camera_keys = get_registered_camera_keys(env)
    video_recorder = EpisodeVideoRecorder(
        output_dir=Path(args_cli.save_replay_video_dir),
        camera_keys=camera_keys,
        fps=args_cli.step_hz,
        enabled=not args_cli.disable_replay_video,
    )
    env.reset(seed=args_cli.seed)
    default_reset_state = copy.deepcopy(env.scene.get_state(is_relative=True))

    print(
        f"[Replay] task={args_cli.task}, task_type={task_type}, action_dim={action_dim}, "
        f"override_initial_robot_state={args_cli.override_initial_robot_state}"
    )
    print(f"[Replay] dataset={args_cli.dataset_file}, selected_episodes={episode_indices}")

    replayed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt):
        with torch.inference_mode():
            for episode_index in episode_indices:
                if not simulation_app.is_running() or simulation_app.is_exiting():
                    break

                episode_name = episode_names[episode_index]
                episode = dataset_file_handler.load_episode(episode_name, env.device)
                if episode is None:
                    raise ValueError(f"Failed to load episode {episode_name}")

                initial_state = episode.get_initial_state()
                actions = episode.data.get("actions")
                if actions is None:
                    raise KeyError(f"Episode {episode_name} is missing 'actions'")
                if actions.shape[-1] != action_dim:
                    raise ValueError(
                        f"Episode {episode_name} action dim {actions.shape[-1]} does not match env action dim {action_dim}"
                    )

                replay_reset_state = build_episode_reset_state(
                    default_reset_state,
                    initial_state,
                    override_initial_robot_state=args_cli.override_initial_robot_state,
                )
                obs_dict, _ = env.reset_to(
                    replay_reset_state,
                    None,
                    seed=int(episode.seed) if episode.seed is not None else None,
                    is_relative=True,
                )
                replayed_episode_count += 1
                print(f"[Replay] episode={episode_index} ({episode_name}), steps={actions.shape[0]}")
                video_recorder.start_episode(episode_name)
                video_recorder.add_observation(obs_dict["policy"], timestamp=time.time())

                while simulation_app.is_running() and not simulation_app.is_exiting():
                    action = episode.get_next_action()
                    if action is None:
                        video_recorder.close(status="completed")
                        break
                    action = action.reshape(1, -1).to(env.device)
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    obs_dict, _, _, _, _ = env.step(action)
                    video_recorder.add_observation(obs_dict["policy"], timestamp=time.time())
                    rate_limiter.sleep(env)
                if not simulation_app.is_running() or simulation_app.is_exiting():
                    video_recorder.close(status="stopped")
                    break

    video_recorder.close(status="stopped")
    print(f"[Replay] Finished replaying {replayed_episode_count} episode{'s' if replayed_episode_count != 1 else ''}.")
    dataset_file_handler.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
