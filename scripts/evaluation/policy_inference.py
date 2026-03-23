"""Script to run a leisaac inference with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
from math import ceil, sqrt
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac inference for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--task_type", type=str, default=None, help="Optional teleop/policy task type override.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help=(
        "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
        " reset."
    ),
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="gr00tn1.5",
    help="Type of policy to use. support gr00tn1.5, gr00tn1.6, lerobot-<model_type>, openpi",
)
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")
parser.add_argument(
    "--save_eval_video_dir",
    type=str,
    default="leisaac_outputs/policy_inference_videos",
    help="Directory to save per-evaluation-round camera videos.",
)
parser.add_argument(
    "--disable_eval_video",
    action="store_true",
    help="Disable per-evaluation-round camera video dumps.",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import time

import carb
import gymnasium as gym
import numpy as np
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)

import leisaac  # noqa: F401


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
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
        """Release the keyboard interface."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True


def get_registered_camera_keys(env: ManagerBasedRLEnv) -> list[str]:
    from isaaclab.sensors import Camera

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

    def start_episode(self, episode_idx: int):
        if not self.enabled:
            return
        self.close(status="interrupted")
        self._current_episode = episode_idx
        self._pending_path = self.output_dir / f"episode_{episode_idx:04d}_pending.mp4"
        self._frame_count = 0
        self._last_frame = None
        self._last_timestamp = None

    def add_frames(self, camera_frames: dict[str, np.ndarray], timestamp: float | None = None):
        if not self.enabled or self._current_episode is None or len(camera_frames) == 0:
            return
        if timestamp is None:
            timestamp = time.time()
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

    def add_observation(self, observation_dict: dict, timestamp: float | None = None):
        if not self.enabled or self._current_episode is None:
            return
        camera_frames = {
            camera_key: observation_dict[camera_key]
            for camera_key in self.camera_keys
            if camera_key in observation_dict
        }
        self.add_frames(camera_frames, timestamp=timestamp)

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
            final_path = self.output_dir / f"episode_{self._current_episode:04d}_{status}.mp4"
            self._pending_path.replace(final_path)
            print(
                f"[Evaluation] Saved episode {self._current_episode} video to {final_path}"
                f" ({self._frame_count} frames, status={status})."
            )
        elif self._frame_count == 0:
            print(f"[Evaluation] Skipped video for episode {self._current_episode}: no camera frames were sent.")

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


def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    """Preprocess the observation dictionary to the format expected by the policy."""
    if model_type in ["gr00tn1.5", "gr00tn1.6", "lerobot", "openpi"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    else:
        raise ValueError(f"Model type {model_type} not supported")


def build_policy(env: ManagerBasedRLEnv, task_type: str):
    from isaaclab.sensors import Camera

    camera_keys = get_registered_camera_keys(env)
    camera_infos = {key: sensor.image_shape for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)}

    if args_cli.policy_type == "gr00tn1.5":
        from leisaac.policy import Gr00tServicePolicyClient

        modality_keys = ["arm_joints", "gripper"] if task_type == "piperleader" else ["single_arm", "gripper"]
        return Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=camera_keys,
            modality_keys=modality_keys,
            task_type=task_type,
        )

    if args_cli.policy_type == "gr00tn1.6":
        from leisaac.policy import Gr00t16ServicePolicyClient

        modality_keys = ["arm_joints", "gripper"] if task_type == "piperleader" else ["single_arm", "gripper"]
        return Gr00t16ServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=camera_keys,
            modality_keys=modality_keys,
            task_type=task_type,
        )

    if "lerobot" in args_cli.policy_type:
        from leisaac.policy import LeRobotServicePolicyClient

        policy_type = args_cli.policy_type.split("-", maxsplit=1)[1]
        return LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos=camera_infos,
            task_type=task_type,
            policy_type=policy_type,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
        )

    if args_cli.policy_type == "openpi":
        from leisaac.policy import OpenPIServicePolicyClient

        return OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=camera_keys,
            task_type=task_type,
        )

    raise ValueError(f"Unsupported policy type: {args_cli.policy_type}")


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task, task_type=args_cli.task_type)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s

    # modify configuration
    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # create policy
    model_type = "lerobot" if "lerobot" in args_cli.policy_type else args_cli.policy_type
    policy = build_policy(env, task_type)
    camera_keys = get_registered_camera_keys(env)
    video_recorder = EpisodeVideoRecorder(
        output_dir=Path(args_cli.save_eval_video_dir),
        camera_keys=camera_keys,
        fps=args_cli.step_hz,
        enabled=not args_cli.disable_eval_video,
    )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()

    # reset environment
    obs_dict, _ = env.reset()
    controller.reset()

    # record the results
    success_count, episode_count = 0, 1

    # simulate environment
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        video_recorder.start_episode(episode_count)
        video_recorder.add_observation(obs_dict["policy"], timestamp=time.time())
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        success, time_out = False, False
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if controller.reset_state:
                    video_recorder.close(status="manual_reset")
                    controller.reset()
                    obs_dict, _ = env.reset()
                    episode_count += 1
                    break

                obs_dict = preprocess_obs_dict(obs_dict["policy"], model_type, args_cli.policy_language_instruction)
                actions = policy.get_action(obs_dict).to(env.device)
                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    video_recorder.add_observation(obs_dict["policy"], timestamp=time.time())
                    if reset_terminated[0]:
                        success = True
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)
            if success or time_out:
                break
        if not simulation_app.is_running():
            video_recorder.close(status="stopped")
            break
        if success:
            video_recorder.close(status="success")
            print(f"[Evaluation] Episode {episode_count} is successful!")
            episode_count += 1
            success_count += 1
        if time_out:
            video_recorder.close(status="time_out")
            print(f"[Evaluation] Episode {episode_count} timed out!")
            episode_count += 1

        finished_episode_count = episode_count - 1
        if finished_episode_count > 0:
            print(
                f"[Evaluation] now success rate: {success_count / finished_episode_count} "
                f" [{success_count}/{finished_episode_count}]"
            )
    if max_episode_count > 0:
        print(
            f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
            f" [{success_count}/{max_episode_count}]"
        )
    else:
        print(f"[Evaluation] Episodes finished: {episode_count - 1}, successes: {success_count}")

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
