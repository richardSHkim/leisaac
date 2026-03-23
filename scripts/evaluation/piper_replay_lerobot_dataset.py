"""Replay a converted LeRobot PiPER IsaacLab HDF5 dataset in simulation."""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import contextlib
import os
import time

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

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
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


def sort_episode_names(episode_names: list[str]) -> list[str]:
    def sort_key(name: str):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name

    return sorted(episode_names, key=sort_key)


def build_episode_reset_state(
    env: ManagerBasedRLEnv,
    episode_initial_state: dict | None,
    override_initial_robot_state: bool,
):
    reset_state = env.scene.get_state(is_relative=True)
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

                env.reset(seed=int(episode.seed) if episode.seed is not None else args_cli.seed)
                replay_reset_state = build_episode_reset_state(
                    env,
                    initial_state,
                    override_initial_robot_state=args_cli.override_initial_robot_state,
                )
                env.reset_to(
                    replay_reset_state,
                    None,
                    seed=int(episode.seed) if episode.seed is not None else None,
                    is_relative=True,
                )
                replayed_episode_count += 1
                print(f"[Replay] episode={episode_index} ({episode_name}), steps={actions.shape[0]}")

                while simulation_app.is_running() and not simulation_app.is_exiting():
                    action = episode.get_next_action()
                    if action is None:
                        break
                    action = action.reshape(1, -1).to(env.device)
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    env.step(action)
                    rate_limiter.sleep(env)

    print(f"[Replay] Finished replaying {replayed_episode_count} episode{'s' if replayed_episode_count != 1 else ''}.")
    dataset_file_handler.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
