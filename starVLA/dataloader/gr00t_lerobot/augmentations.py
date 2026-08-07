"""Deterministic, geometry-aware dataset augmentations.

The EgoS2 camera is an OpenCV camera frame (``+x`` points to image-right).
Reflecting an example therefore requires more than flipping its image: the
left/right action streams must be exchanged and the reflected vector/rotation
components must be transformed as well.

This module deliberately contains only numpy operations so it can be used in
dataset workers before the normalisation transforms run.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

import numpy as np


_ADAMU_STATE_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("shoulderPitch_Left", "shoulderPitch_Right", 1.0),
    ("shoulderRoll_Left", "shoulderRoll_Right", -1.0),
    ("shoulderYaw_Left", "shoulderYaw_Right", -1.0),
    ("elbow_Left", "elbow_Right", 1.0),
    ("wristYaw_Left", "wristYaw_Right", -1.0),
    ("wristPitch_Left", "wristPitch_Right", 1.0),
    ("wristRoll_Left", "wristRoll_Right", -1.0),
)
_ADAMU_HAND_PAIRS: tuple[tuple[str, str, float], ...] = tuple(
    (f"L_{suffix}", f"R_{suffix}", 1.0)
    for suffix in (
        "thumb_MCP_joint1",
        "thumb_MCP_joint2",
        "thumb_PIP_joint",
        "thumb_DIP_joint",
        "index_MCP_joint",
        "index_DIP_joint",
        "middle_MCP_joint",
        "middle_DIP_joint",
        "ring_MCP_joint",
        "ring_DIP_joint",
        "pinky_MCP_joint",
        "pinky_DIP_joint",
    )
)
_ADAMU_CENTER_SIGNS: tuple[tuple[str, float], ...] = (
    ("waistYaw", -1.0),
    ("waistRoll", -1.0),
    ("waistPitch", 1.0),
    ("neckYaw", -1.0),
    ("neckPitch", 1.0),
)

_SIDE_WORD_PATTERN = re.compile(r"(?<![A-Za-z])(left|right)(?![A-Za-z])", re.IGNORECASE)


def deterministic_mirror_coin(
    probability: float,
    *,
    seed: int,
    epoch: int,
    sample_index: int | None,
    trajectory_id: int | str | None,
    base_index: int | None,
) -> bool:
    """Return a reproducible Bernoulli decision for one dataset sample.

    Using a hash instead of process-global RNG state keeps the 50% decision
    stable across dataloader workers and makes a run reproducible.  The epoch
    is part of the key, so an example can receive either view on a later pass.
    """

    probability = float(probability)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True

    key = repr(
        (
            int(seed),
            int(epoch),
            None if sample_index is None else int(sample_index),
            None if trajectory_id is None else str(trajectory_id),
            None if base_index is None else int(base_index),
        )
    ).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    random_value = int.from_bytes(digest, byteorder="little") / float(2**64)
    return random_value < probability


def mirror_egos2_instruction(instruction: str) -> str:
    """Swap explicit left/right words in an instruction for a mirrored view.

    The source LeRobot task text remains unchanged.  This conversion is only
    applied after the per-sample mirror coin has selected the reflected view.
    Replacement is simultaneous so ``left`` and ``right`` do not get changed
    twice, and common capitalization is preserved.
    """

    if not isinstance(instruction, str):
        raise TypeError(f"Expected instruction text, got {type(instruction).__name__}")

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        is_left = value.casefold() == "left"
        replacement = "right" if is_left else "left"
        if value.isupper():
            return replacement.upper()
        if value[:1].isupper():
            return replacement.capitalize()
        return replacement

    return _SIDE_WORD_PATTERN.sub(replace, instruction)


def _copy_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Expected a numeric array for mirroring, got {array.dtype}")
    return array.copy()


def _reflect_video(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 3:
        raise ValueError(f"Expected an image/video array with at least 3 dimensions, got {array.shape}")
    # HWC or THWC: the image-width axis is the second-last axis.
    return np.flip(array, axis=-2).copy()


def _swap_reflected(
    data: Mapping[str, Any],
    output: dict[str, Any],
    left_key: str,
    right_key: str,
    signs: Sequence[float],
) -> None:
    if left_key not in data or right_key not in data:
        raise KeyError(f"EgoS2 mirror pair is incomplete: {left_key}, {right_key}")
    left = _copy_array(data[left_key])
    right = _copy_array(data[right_key])
    signs_array = np.asarray(signs, dtype=left.dtype)
    if left.shape[-1] != len(signs_array) or right.shape[-1] != len(signs_array):
        raise ValueError(
            f"Mirror signs for {left_key}/{right_key} have length {len(signs_array)}, "
            f"but arrays have shapes {left.shape} and {right.shape}"
        )
    output[left_key] = right * signs_array
    output[right_key] = left * signs_array


def mirror_egos2_action(data: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the named EgoS2 action fields in a raw sample dictionary."""

    output = dict(data)
    _swap_reflected(
        data,
        output,
        "action.left_eef_delta_position",
        "action.right_eef_delta_position",
        (-1.0, 1.0, 1.0),
    )
    _swap_reflected(
        data,
        output,
        "action.left_eef_delta_quaternion",
        "action.right_eef_delta_quaternion",
        (1.0, 1.0, -1.0, -1.0),
    )
    _swap_reflected(
        data,
        output,
        "action.left_hand",
        "action.right_hand",
        (1.0,) * 11,
    )
    return output


def mirror_egos2_state(values: Any, names: Sequence[str]) -> np.ndarray:
    """Reflect a legacy 43-D Adamu qpos state by joint name.

    New StarVLA datasets use :func:`mirror_egos2_eef_pose` instead.  This
    compatibility path remains useful for older exported qpos archives.
    """

    output = _copy_array(values)
    lookup = {str(name): index for index, name in enumerate(names)}
    if len(lookup) != len(names):
        raise ValueError("observation.state names contain duplicates")
    if output.shape[-1] != len(names):
        raise ValueError(f"State shape {output.shape} does not match {len(names)} names")

    pairs = _ADAMU_STATE_PAIRS + _ADAMU_HAND_PAIRS
    for left_name, right_name, sign in pairs:
        if left_name not in lookup or right_name not in lookup:
            raise KeyError(f"EgoS2 state mirror pair is incomplete: {left_name}, {right_name}")
        left = output[..., lookup[left_name]].copy()
        right = output[..., lookup[right_name]].copy()
        output[..., lookup[left_name]] = sign * right
        output[..., lookup[right_name]] = sign * left

    for name, sign in _ADAMU_CENTER_SIGNS:
        if name not in lookup:
            raise KeyError(f"EgoS2 state mirror joint is missing: {name}")
        output[..., lookup[name]] = sign * output[..., lookup[name]]
    return output


def mirror_egos2_eef_pose(values: Any) -> np.ndarray:
    """Reflect the current 36-D absolute camera-pose state.

    Layout per side is ``[position(3), quaternion_wxyz(4), hand(11)]``.
    The state uses absolute EEF pose while the action uses the corresponding
    delta pose, so the geometric reflection is shared by both contracts.
    """

    values = _copy_array(values)
    if values.shape[-1] != 36:
        raise ValueError(f"EgoS2 absolute EEF pose state must end in 36, got {values.shape}")

    output = values.copy()
    left_position = values[..., 0:3]
    left_quaternion = values[..., 3:7]
    left_hand = values[..., 7:18]
    right_position = values[..., 18:21]
    right_quaternion = values[..., 21:25]
    right_hand = values[..., 25:36]

    position_signs = np.asarray((-1.0, 1.0, 1.0), dtype=values.dtype)
    quaternion_signs = np.asarray((1.0, 1.0, -1.0, -1.0), dtype=values.dtype)
    output[..., 0:3] = right_position * position_signs
    output[..., 3:7] = right_quaternion * quaternion_signs
    output[..., 7:18] = right_hand
    output[..., 18:21] = left_position * position_signs
    output[..., 21:25] = left_quaternion * quaternion_signs
    output[..., 25:36] = left_hand
    return output


def mirror_egos2_sample(
    data: Mapping[str, Any],
    *,
    state_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a left/right-reflected raw EgoS2 sample.

    The returned dictionary is a shallow copy; every modified array is copied
    before it is changed.  ``_mirror_applied`` is a private marker consumed by
    the dataset packer and is intentionally ignored by modality transforms.
    """

    output = mirror_egos2_action(data)
    for key, value in list(data.items()):
        if str(key).startswith("video."):
            output[key] = _reflect_video(value)
        elif str(key).startswith("annotation."):
            # get_language() returns a one-element list of task strings.  Do
            # this before the normal modality transforms and before _pack_sample
            # extracts ``lang`` so the model sees the mirrored instruction.
            if isinstance(value, str):
                output[key] = mirror_egos2_instruction(value)
            elif isinstance(value, list):
                output[key] = [mirror_egos2_instruction(item) for item in value]
            elif isinstance(value, tuple):
                output[key] = tuple(mirror_egos2_instruction(item) for item in value)

    pose_state_key = "state.adamu_eef_pose"
    legacy_state_key = "state.adamu_qpos"
    split_state_keys = (
        "state.left_eef_position",
        "state.left_eef_quaternion",
        "state.left_hand",
        "state.right_eef_position",
        "state.right_eef_quaternion",
        "state.right_hand",
    )
    if all(key in data for key in split_state_keys):
        _swap_reflected(
            data,
            output,
            "state.left_eef_position",
            "state.right_eef_position",
            (-1.0, 1.0, 1.0),
        )
        _swap_reflected(
            data,
            output,
            "state.left_eef_quaternion",
            "state.right_eef_quaternion",
            (1.0, 1.0, -1.0, -1.0),
        )
        _swap_reflected(
            data,
            output,
            "state.left_hand",
            "state.right_hand",
            (1.0,) * 11,
        )
    elif pose_state_key in data:
        output[pose_state_key] = mirror_egos2_eef_pose(data[pose_state_key])
    elif legacy_state_key in data:
        if state_names is None:
            raise ValueError("state_names are required to mirror legacy state.adamu_qpos")
        output[legacy_state_key] = mirror_egos2_state(data[legacy_state_key], state_names)

    output["_mirror_applied"] = True
    return output
