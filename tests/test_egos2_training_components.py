import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.augmentations import (
    deterministic_mirror_coin,
    mirror_egos2_sample,
    mirror_egos2_eef_pose,
    mirror_egos2_instruction,
)
from starVLA.dataloader.gr00t_lerobot.datasets import compute_action_valid_mask
from starVLA.dataloader.lerobot_datasets import OverfitSampler
from starVLA.training.trainer_utils.action_loss import compute_masked_action_l1_loss
from starVLA.training.trainer_utils.loss_diagnostics import (
    LossSpikeTracker,
    OverfitAnalyzer,
)


class EgoS2TrainingComponentsTest(unittest.TestCase):
    def test_mirror_reflects_camera_action_and_video(self):
        names = [
            "waistRoll", "waistPitch", "waistYaw",
            "shoulderPitch_Left", "shoulderRoll_Left", "shoulderYaw_Left",
            "elbow_Left", "wristYaw_Left", "wristPitch_Left", "wristRoll_Left",
            "L_thumb_MCP_joint1", "L_thumb_MCP_joint2", "L_thumb_PIP_joint", "L_thumb_DIP_joint",
            "L_index_MCP_joint", "L_index_DIP_joint", "L_middle_MCP_joint", "L_middle_DIP_joint",
            "L_ring_MCP_joint", "L_ring_DIP_joint", "L_pinky_MCP_joint", "L_pinky_DIP_joint",
            "shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right",
            "elbow_Right", "wristYaw_Right", "wristPitch_Right", "wristRoll_Right",
            "R_thumb_MCP_joint1", "R_thumb_MCP_joint2", "R_thumb_PIP_joint", "R_thumb_DIP_joint",
            "R_index_MCP_joint", "R_index_DIP_joint", "R_middle_MCP_joint", "R_middle_DIP_joint",
            "R_ring_MCP_joint", "R_ring_DIP_joint", "R_pinky_MCP_joint", "R_pinky_DIP_joint",
            "neckYaw", "neckPitch",
        ]
        data = {
            "action.left_eef_delta_position": np.array([[1, 2, 3]], dtype=np.float32),
            "action.right_eef_delta_position": np.array([[4, 5, 6]], dtype=np.float32),
            "action.left_eef_delta_quaternion": np.array([[.5, .5, .5, .5]], dtype=np.float32),
            "action.right_eef_delta_quaternion": np.array([[.5, -.5, .5, -.5]], dtype=np.float32),
            "action.left_hand": np.arange(11, dtype=np.float32)[None],
            "action.right_hand": (100 + np.arange(11, dtype=np.float32))[None],
            "state.adamu_qpos": np.arange(43, dtype=np.float32)[None],
            "video.camera": np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3),
        }
        mirrored = mirror_egos2_sample(data, state_names=names)
        np.testing.assert_allclose(mirrored["action.left_eef_delta_position"], [[-4, 5, 6]])
        np.testing.assert_allclose(mirrored["action.left_eef_delta_quaternion"], [[.5, -.5, -.5, .5]])
        np.testing.assert_array_equal(mirrored["video.camera"], data["video.camera"][:, :, ::-1])
        self.assertTrue(mirrored["_mirror_applied"])

    def test_mirror_swaps_instruction_side_words(self):
        self.assertEqual(
            mirror_egos2_instruction("Pick with the right hand."),
            "Pick with the left hand.",
        )
        self.assertEqual(
            mirror_egos2_instruction("Use the LEFT hand, not the right hand."),
            "Use the RIGHT hand, not the left hand.",
        )
        data = {
            "annotation.human.task_description": ["move right hand"],
            "action.left_eef_delta_position": np.zeros((1, 3), dtype=np.float32),
            "action.right_eef_delta_position": np.zeros((1, 3), dtype=np.float32),
            "action.left_eef_delta_quaternion": np.asarray([[1, 0, 0, 0]], dtype=np.float32),
            "action.right_eef_delta_quaternion": np.asarray([[1, 0, 0, 0]], dtype=np.float32),
            "action.left_hand": np.zeros((1, 11), dtype=np.float32),
            "action.right_hand": np.zeros((1, 11), dtype=np.float32),
        }
        mirrored = mirror_egos2_sample(data)
        self.assertEqual(mirrored["annotation.human.task_description"], ["move left hand"])

    def test_mirror_coin_and_overfit_sampler_are_deterministic(self):
        values = [
            deterministic_mirror_coin(.5, seed=42, epoch=0, sample_index=i, trajectory_id=0, base_index=i)
            for i in range(1000)
        ]
        self.assertGreater(sum(values) / len(values), .45)
        self.assertLess(sum(values) / len(values), .55)
        sampler = OverfitSampler(100, 3)
        self.assertEqual(list(sampler), [0, 1, 2])

    def test_mirror_reflects_36d_absolute_eef_pose_state(self):
        state = np.arange(36, dtype=np.float32)[None]
        mirrored = mirror_egos2_eef_pose(state)
        np.testing.assert_allclose(mirrored[0, 0:3], [-18, 19, 20])
        np.testing.assert_allclose(mirrored[0, 3:7], [21, 22, -23, -24])
        np.testing.assert_array_equal(mirrored[0, 7:18], state[0, 25:36])
        np.testing.assert_allclose(mirrored[0, 18:21], [-0, 1, 2])
        np.testing.assert_allclose(mirrored[0, 21:25], [3, 4, -5, -6])
        np.testing.assert_array_equal(mirrored[0, 25:36], state[0, 7:18])

    def test_action_valid_mask_only_marks_boundary_padding(self):
        np.testing.assert_array_equal(
            compute_action_valid_mask(
                base_index=85,
                trajectory_length=91,
                action_indices=list(range(8)),
            ),
            [1, 1, 1, 1, 1, 1, 0, 0],
        )
        np.testing.assert_array_equal(
            compute_action_valid_mask(
                base_index=90,
                trajectory_length=91,
                action_indices=list(range(8)),
            ),
            [1, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_masked_loss_ignores_only_padding_positions(self):
        predictions = torch.tensor([[[1.0], [3.0], [100.0]]])
        targets = torch.zeros_like(predictions)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        loss = compute_masked_action_l1_loss(predictions, targets, mask)
        self.assertAlmostEqual(loss.item(), 2.0)

    def test_loss_spike_and_overfit_reports_include_sample_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = [{
                "sample_index": 7,
                "sample_id": "lerobot_v21/episode_000003/step_9",
                "dataset_name": "lerobot_v21",
                "trajectory_id": 3,
                "base_index": 9,
            }]
            tracker = LossSpikeTracker(
                root,
                {"enabled": True, "warmup_steps": 1, "ratio_threshold": 3, "ema_decay": .5},
            )
            for step, value in enumerate((1., 1., 10.), 1):
                tracker.update(
                    step=step,
                    micro_step=step,
                    batch=batch,
                    per_sample_losses=torch.tensor([value]),
                )
            record = json.loads((root / "loss_spikes_rank0.jsonl").read_text().splitlines()[0])
            self.assertEqual(record["sample_index"], 7)
            self.assertEqual(record["base_index"], 9)

            analyzer = OverfitAnalyzer(root, {"enabled": True, "target_ratio": .2})
            for step, value in enumerate((1., .4, .1), 1):
                analyzer.update(
                    step=step,
                    micro_step=step,
                    batch=batch,
                    per_sample_losses=torch.tensor([value]),
                )
            report = json.loads(analyzer.finalize().read_text())
            self.assertEqual(report["successful_samples"], 1)
            self.assertEqual(report["samples"][0]["sample_id"], batch[0]["sample_id"])


if __name__ == "__main__":
    unittest.main()
