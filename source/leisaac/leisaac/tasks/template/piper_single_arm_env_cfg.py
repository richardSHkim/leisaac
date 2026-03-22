from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg as RecordTerm,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets.episode_data import EpisodeData
from leisaac.assets.robots.lerobot import PIPER_FOLLOWER_CFG
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot

from . import mdp


PIPER_ENV_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
PIPER_FEATURE_JOINT_NAMES = [
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "gripper.pos",
]
PIPER_ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=PIPER_ENV_JOINT_NAMES, preserve_order=True)
PIPER_ARM_BASE_PRIM_PATH = "{ENV_REGEX_NS}/Robot/arm_base"
PIPER_LINK6_PRIM_PATH = "{ENV_REGEX_NS}/Robot/link6"
PIPER_CAMERA_WIDTH = 640
PIPER_CAMERA_HEIGHT = 480
PIPER_WRIST_CAMERA_INTRINSICS = [
    605.390869140625,
    0.0,
    321.57623291015625,
    0.0,
    605.4684448242188,
    253.18133544921875,
    0.0,
    0.0,
    1.0,
]
PIPER_BASE_CAMERA_INTRINSICS = [
    605.8900146484375,
    0.0,
    326.1027526855469,
    0.0,
    605.5436401367188,
    243.02032470703125,
    0.0,
    0.0,
    1.0,
]


@configclass
class PiperSingleArmTaskSceneCfg(InteractiveSceneCfg):
    """Scene configuration for PiPER single-arm tasks."""

    scene: AssetBaseCfg = MISSING

    robot: ArticulationCfg = PIPER_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path=PIPER_ARM_BASE_PRIM_PATH,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=PIPER_LINK6_PRIM_PATH,
                name="gripper",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=PIPER_LINK6_PRIM_PATH,
                name="jaw",
                offset=OffsetCfg(pos=(-0.01, 0.06, 0.14)),
            ),
        ],
    )

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"{PIPER_LINK6_PRIM_PATH}/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0, 0.065, 0.05), rot=(0.0, 0.0, -0.173648, 0.984808), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=PIPER_WRIST_CAMERA_INTRINSICS,
            width=PIPER_CAMERA_WIDTH,
            height=PIPER_CAMERA_HEIGHT,
            focus_distance=400.0,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=PIPER_CAMERA_WIDTH,
        height=PIPER_CAMERA_HEIGHT,
        update_period=1 / 30.0,
    )

    base: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"{PIPER_ARM_BASE_PRIM_PATH}/base_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.55, -0.7, 0.42), rot=(0.77337, 0.55078, -0.2374, -0.20537), convention="opengl"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=PIPER_BASE_CAMERA_INTRINSICS,
            width=PIPER_CAMERA_WIDTH,
            height=PIPER_CAMERA_HEIGHT,
            focus_distance=400.0,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=PIPER_CAMERA_WIDTH,
        height=PIPER_CAMERA_HEIGHT,
        update_period=1 / 30.0,
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class PiperSingleArmActionsCfg:
    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class PiperSingleArmEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class PiperSingleArmObservationsCfg:
    """Observation specifications for PiPER tasks."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": PIPER_ROBOT_JOINT_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": PIPER_ROBOT_JOINT_CFG})
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": PIPER_ROBOT_JOINT_CFG}
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": PIPER_ROBOT_JOINT_CFG}
        )
        actions = ObsTerm(func=mdp.last_action)
        wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False}
        )
        base = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("base"), "data_type": "rgb", "normalize": False}
        )
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target,
            params={"asset_cfg": PIPER_ROBOT_JOINT_CFG},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class PiperSingleArmRewardsCfg:
    """Configuration for rewards."""


@configclass
class PiperSingleArmTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PiperSingleArmTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for PiPER single-arm task template environment."""

    scene: PiperSingleArmTaskSceneCfg = MISSING

    observations: PiperSingleArmObservationsCfg = MISSING
    actions: PiperSingleArmActionsCfg = PiperSingleArmActionsCfg()
    events: PiperSingleArmEventCfg = PiperSingleArmEventCfg()

    rewards: PiperSingleArmRewardsCfg = PiperSingleArmRewardsCfg()
    terminations: PiperSingleArmTerminationsCfg = MISSING

    recorders: RecordTerm = RecordTerm()

    dynamic_reset_gripper_effort_limit: bool = True
    robot_name: str = "piper_follower"
    default_feature_joint_names: list[str] = MISSING
    task_description: str = MISSING

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 1
        self.episode_length_s = 25.0
        self.viewer.eye = (-0.6, -0.9, 0.8)
        self.viewer.lookat = (0.5, -0.1, 0.15)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

        self.task_type = "piperleader"
        self.default_feature_joint_names = PIPER_FEATURE_JOINT_NAMES

    def use_teleop_device(self, teleop_device) -> None:
        self.task_type = teleop_device
        self.actions = init_action_cfg(self.actions, device=teleop_device)

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def build_lerobot_frame(self, episode_data: EpisodeData, dataset_cfg: LeRobotDatasetCfg) -> dict:
        obs_data = episode_data._data["obs"]
        action = obs_data["actions"][-1]
        if dataset_cfg.action_align:
            processed_action = convert_leisaac_action_to_lerobot(
                action.unsqueeze(0), task_type=self.task_type, robot_name=self.robot_name
            ).squeeze(0)
        else:
            processed_action = action.cpu().numpy()
        frame = {
            "action": processed_action,
            "observation.state": convert_leisaac_action_to_lerobot(
                obs_data["joint_pos"][-1].unsqueeze(0), task_type=self.task_type, robot_name=self.robot_name
            ).squeeze(0),
            "task": self.task_description,
        }
        for frame_key in dataset_cfg.features.keys():
            if not frame_key.startswith("observation.images"):
                continue
            camera_key = frame_key.split(".")[-1]
            frame[frame_key] = obs_data[camera_key][-1].cpu().numpy()

        return frame
