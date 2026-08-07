"""EgoS2 Adamu StarVLA data registry.

The dataset stores one flat ``observation.state`` vector and one flat
``action`` vector. ``modality.json`` gives the StarVLA/GR00T loader the named
slices; this registry supplies the corresponding transforms and mixture.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class EgoS2AdamuDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = ["video.camera"]
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_quaternion",
        "state.left_hand",
        "state.right_eef_position",
        "state.right_eef_quaternion",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_eef_delta_position",
        "action.left_eef_delta_quaternion",
        "action.left_hand",
        "action.right_eef_delta_position",
        "action.right_eef_delta_quaternion",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    state_key_dims = {
        "state.left_eef_position": 3,
        "state.left_eef_quaternion": 4,
        "state.left_hand": 11,
        "state.right_eef_position": 3,
        "state.right_eef_quaternion": 4,
        "state.right_hand": 11,
    }
    action_key_dims = {
        "action.left_eef_delta_position": 3,
        "action.left_eef_delta_quaternion": 4,
        "action.left_hand": 11,
        "action.right_eef_delta_position": 3,
        "action.right_eef_delta_quaternion": 4,
        "action.right_hand": 11,
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        state_normalization = {key: "min_max" for key in self.state_keys}
        for key in (
            "state.left_eef_quaternion",
            "state.right_eef_quaternion",
        ):
            state_normalization[key] = "identity"
        action_normalization = {key: "min_max" for key in self.action_keys}
        for key in (
            "action.left_eef_delta_quaternion",
            "action.right_eef_delta_quaternion",
        ):
            action_normalization[key] = "identity"
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=state_normalization,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_normalization,
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "EgoS2_Adamu": EgoS2AdamuDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "EgoS2_Adamu_Open": [
        ("lerobot_v21", 1.0, "EgoS2_Adamu"),
    ],
}
