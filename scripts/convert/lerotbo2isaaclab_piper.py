"""Convert a LeRobot v3 PiPER dataset into an IsaacLab-compatible replay HDF5 file."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Disable HDF5 file locking for shared filesystems.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

PIPER_ROOT_POSE = np.array([[0.35, -0.64, 0.01, 0.70710678, 0.0, 0.0, 0.70710678]], dtype=np.float32)
PIPER_ROOT_VELOCITY = np.zeros((1, 6), dtype=np.float32)
PIPER_JOINT_VELOCITY = np.zeros((1, 8), dtype=np.float32)
PIPER_FOLLOWER_USD_JOINT_LIMITS = {
    "joint_1": (-2.618, 2.618),
    "joint_2": (0.0, 3.14),
    "joint_3": (-2.697, 0.0),
    "joint_4": (-1.832, 1.832),
    "joint_5": (-1.22, 1.22),
    "joint_6": (-3.14, 3.14),
    "gripper": (0.0, 0.05),
}
PIPER_FOLLOWER_MOTOR_LIMITS = {
    "joint_1": (-2.618, 2.618),
    "joint_2": (0.0, 3.14),
    "joint_3": (-2.697, 0.0),
    "joint_4": (-1.832, 1.832),
    "joint_5": (-1.22, 1.22),
    "joint_6": (-3.14, 3.14),
    "gripper": (0.0, 1.0),
}

try:
    from leisaac.utils.robot_utils import convert_lerobot_action_to_leisaac as _shared_convert_lerobot_action_to_leisaac
except Exception:
    _shared_convert_lerobot_action_to_leisaac = None


def _json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _require_parquet_backend():
    try:
        import pyarrow.parquet as pq  # noqa: F401

        return "pyarrow"
    except ImportError:
        pass

    try:
        import pandas as pd  # noqa: F401

        return "pandas"
    except ImportError as exc:
        raise ImportError(
            "Direct LeRobot v3 conversion requires a parquet reader. Install `pyarrow` "
            "or a pandas parquet backend in the runtime used for this script."
        ) from exc


def _iter_parquet_rows(parquet_path: Path, columns: list[str]) -> list[dict[str, Any]]:
    backend = _require_parquet_backend()
    read_columns = columns or None

    if backend == "pyarrow":
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path, columns=read_columns)
        data = table.to_pydict()
        rows = []
        for row_index in range(table.num_rows):
            rows.append({column: data[column][row_index] for column in data})
        return rows

    import pandas as pd

    try:
        dataframe = pd.read_parquet(parquet_path, columns=read_columns)
    except Exception as exc:
        raise ImportError(
            "Failed to read parquet data. Install `pyarrow` or a working pandas parquet backend "
            "in the runtime used for this script."
        ) from exc
    rows = []
    for record in dataframe.to_dict(orient="records"):
        rows.append(record)
    return rows


def _to_scalar_int(value: Any, field_name: str) -> int:
    scalar = np.asarray(value).reshape(-1)
    if scalar.size != 1:
        raise ValueError(f"Expected scalar field for {field_name}, got shape {np.asarray(value).shape}")
    return int(scalar[0])


def _to_float_vector(value: Any, field_name: str, expected_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] != expected_dim:
        raise ValueError(f"Expected {field_name} to have dim {expected_dim}, got shape {np.asarray(value).shape}")
    return array


def _convert_lerobot_action_to_leisaac_local(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    processed_action = np.zeros_like(action)
    for index, joint_name in enumerate(PIPER_FOLLOWER_USD_JOINT_LIMITS):
        motor_limit_range = PIPER_FOLLOWER_MOTOR_LIMITS[joint_name]
        joint_limit_range = PIPER_FOLLOWER_USD_JOINT_LIMITS[joint_name]
        motor_range = motor_limit_range[1] - motor_limit_range[0]
        joint_range = joint_limit_range[1] - joint_limit_range[0]
        motor_value = action[:, index] - motor_limit_range[0]
        processed_action[:, index] = motor_value / motor_range * joint_range + joint_limit_range[0]
    return processed_action.astype(np.float32)


def convert_lerobot_action_to_leisaac(action: np.ndarray, task_type: str, robot_name: str) -> np.ndarray:
    if _shared_convert_lerobot_action_to_leisaac is not None:
        return _shared_convert_lerobot_action_to_leisaac(
            action,
            task_type=task_type,
            robot_name=robot_name,
        ).astype(np.float32)
    return _convert_lerobot_action_to_leisaac_local(action)


def expand_piper_joint_state(single_gripper_joint_state: np.ndarray) -> np.ndarray:
    if single_gripper_joint_state.shape[-1] != 7:
        raise ValueError(f"Expected PiPER joint state dim 7, got {single_gripper_joint_state.shape}")
    expanded = np.zeros((single_gripper_joint_state.shape[0], 8), dtype=np.float32)
    expanded[:, :6] = single_gripper_joint_state[:, :6]
    expanded[:, 6] = single_gripper_joint_state[:, 6]
    expanded[:, 7] = -single_gripper_joint_state[:, 6]
    return expanded


def build_episode_records(dataset_root: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata file: {info_path}")

    info = _read_json(info_path)
    features = info.get("features", {})
    required_features = {"action", "observation.state"}
    missing = required_features - set(features.keys())
    if missing:
        raise ValueError(f"Dataset is missing required features: {sorted(missing)}")
    if info.get("robot_type") != "piper_follower":
        raise ValueError(f"Expected robot_type='piper_follower', got {info.get('robot_type')!r}")

    parquet_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    columns = ["action", "observation.state", "episode_index", "frame_index", "timestamp", "task_index"]
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    global_order = 0
    for parquet_path in parquet_files:
        rows = _iter_parquet_rows(parquet_path, columns)
        for row in rows:
            episode_index = _to_scalar_int(row["episode_index"], "episode_index")
            frame_index = _to_scalar_int(row["frame_index"], "frame_index") if "frame_index" in row else global_order
            task_index = _to_scalar_int(row["task_index"], "task_index") if "task_index" in row else -1
            timestamp = float(np.asarray(row["timestamp"]).reshape(-1)[0]) if "timestamp" in row else float(global_order)
            episodes[episode_index].append(
                {
                    "frame_index": frame_index,
                    "global_order": global_order,
                    "timestamp": timestamp,
                    "task_index": task_index,
                    "action": _to_float_vector(row["action"], "action", 7),
                    "observation_state": _to_float_vector(row["observation.state"], "observation.state", 7),
                }
            )
            global_order += 1

    tasks_rows: list[dict[str, Any]] = []
    tasks_parquet = dataset_root / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        for row in _iter_parquet_rows(tasks_parquet, []):
            tasks_rows.append(row)

    return episodes, info, tasks_rows


def _write_nested_dataset(group: h5py.Group, key_path: str, value: np.ndarray):
    current_group = group
    sub_keys = key_path.split("/")
    for sub_key in sub_keys[:-1]:
        current_group = current_group.require_group(sub_key)
    current_group.create_dataset(sub_keys[-1], data=value, compression="gzip")


def _write_string_dataset(group: h5py.Group, dataset_name: str, text: str):
    group.create_dataset(dataset_name, data=text, dtype=h5py.string_dtype(encoding="utf-8"))


def convert_dataset(
    lerobot_root: Path,
    output_hdf5: Path,
    task: str | None,
    task_type: str,
    selected_episodes: list[int] | None,
):
    episodes, info, tasks_rows = build_episode_records(lerobot_root)
    available_episode_indices = sorted(episodes.keys())
    if not available_episode_indices:
        raise ValueError(f"No episodes found in dataset {lerobot_root}")

    episode_indices = available_episode_indices if not selected_episodes else selected_episodes
    missing_episode_indices = [index for index in episode_indices if index not in episodes]
    if missing_episode_indices:
        raise ValueError(f"Requested episodes are missing from dataset: {missing_episode_indices}")

    if output_hdf5.exists():
        raise FileExistsError(f"Output file already exists: {output_hdf5}")
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_hdf5, "w") as output_file:
        output_file.attrs["created_at"] = dt.datetime.now(dt.UTC).isoformat()
        output_file.attrs["source_lerobot_root"] = str(lerobot_root)
        output_file.attrs["source_codebase_version"] = info.get("codebase_version", "")
        output_file.attrs["task_type"] = task_type
        if task is not None:
            output_file.attrs["task"] = task

        meta_group = output_file.create_group("meta")
        _write_string_dataset(meta_group, "info.json", json.dumps(info, indent=2, default=_json_default))
        _write_string_dataset(meta_group, "tasks.json", json.dumps(tasks_rows, indent=2, default=_json_default))

        data_group = output_file.create_group("data")
        data_group.attrs["total"] = 0
        data_group.attrs["env_args"] = json.dumps({"env_name": task or "", "type": 2})

        for episode_index in episode_indices:
            frames = sorted(episodes[episode_index], key=lambda item: (item["frame_index"], item["global_order"]))
            if not frames:
                continue

            lerobot_actions = np.stack([frame["action"] for frame in frames], axis=0)
            lerobot_joint_states = np.stack([frame["observation_state"] for frame in frames], axis=0)
            isaaclab_actions = convert_lerobot_action_to_leisaac(
                lerobot_actions,
                task_type=task_type,
                robot_name="piper_follower",
            )
            isaaclab_joint_state = convert_lerobot_action_to_leisaac(
                lerobot_joint_states[:1],
                task_type=task_type,
                robot_name="piper_follower",
            )
            initial_joint_position = expand_piper_joint_state(isaaclab_joint_state)

            episode_group = data_group.create_group(f"demo_{episode_index}")
            episode_group.attrs["num_samples"] = int(isaaclab_actions.shape[0])
            episode_group.attrs["source_episode_index"] = episode_index
            episode_group.attrs["task_index"] = frames[0]["task_index"]
            episode_group.attrs["source_start_timestamp"] = frames[0]["timestamp"]
            episode_group.attrs["source_end_timestamp"] = frames[-1]["timestamp"]

            _write_nested_dataset(
                episode_group,
                "initial_state/articulation/robot/root_pose",
                PIPER_ROOT_POSE,
            )
            _write_nested_dataset(
                episode_group,
                "initial_state/articulation/robot/root_velocity",
                PIPER_ROOT_VELOCITY,
            )
            _write_nested_dataset(
                episode_group,
                "initial_state/articulation/robot/joint_position",
                initial_joint_position,
            )
            _write_nested_dataset(
                episode_group,
                "initial_state/articulation/robot/joint_velocity",
                PIPER_JOINT_VELOCITY,
            )
            _write_nested_dataset(episode_group, "actions", isaaclab_actions.astype(np.float32))

            data_group.attrs["total"] += int(isaaclab_actions.shape[0])

    print(f"[Convert] source={lerobot_root}")
    print(f"[Convert] output={output_hdf5}")
    print(f"[Convert] episodes={episode_indices}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LeRobot v3 PiPER dataset to IsaacLab replay HDF5.")
    parser.add_argument(
        "--lerobot_root",
        type=str,
        default="datasets/richardshkim/piper_banana_v2",
        help="Path to the local LeRobot v3 dataset root.",
    )
    parser.add_argument(
        "--output_hdf5",
        type=str,
        required=True,
        help="Path to the output IsaacLab-compatible HDF5 replay file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task name hint to store in HDF5 env metadata. Replay-time task selection is separate.",
    )
    parser.add_argument("--task_type", type=str, default="piperleader", help="Teleop/action task type.")
    parser.add_argument(
        "--select_episodes",
        type=int,
        nargs="+",
        default=[],
        help="Episode indices to convert. Empty means convert all episodes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_dataset(
        lerobot_root=Path(args.lerobot_root).expanduser().resolve(),
        output_hdf5=Path(args.output_hdf5).expanduser().resolve(),
        task=args.task,
        task_type=args.task_type,
        selected_episodes=args.select_episodes,
    )


if __name__ == "__main__":
    main()
