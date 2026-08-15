# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md (adapt a little code)

"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils.action_loss import (
    build_action_valid_mask,
    compute_masked_action_l1_loss,
)
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenOFT
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenOFTDefaultConfig:
    """QwenOFT framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenOFT"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (MLP regression over action special tokens) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "MLP",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # Hidden dim for the action MLP (auto-set from VLM hidden_size at runtime)
            "action_hidden_dim": 2560,
            # How many future steps to predict
            "future_action_window_size": 8,
            # How many past steps included in action chunk (usually 0)
            "past_action_window_size": 0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    Multimodal vision-language-action model (OFT variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Action special token injected into the VLM sequence
      - MLP regression head over action token hidden states (L1 loss)

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenOFTDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align action_hidden_dim to VLM hidden_size at runtime
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        # EgoS2 provides a normalized 36-D proprioceptive state.  The old
        # path only rendered it as 256-bin text, which is lossy for a tiny
        # memorization/regression set and makes neighboring frames nearly
        # indistinguishable to the action head.  Keep the text conditioning
        # for compatibility, and optionally add a small continuous state
        # projection to every action query.  The module is opt-in so old
        # checkpoints without this state_dict branch remain loadable.
        action_model_cfg = self.config.framework.action_model
        state_dim = int(action_model_cfg.get("state_dim", 0) or 0)
        self.use_state_conditioning = bool(
            action_model_cfg.get("use_state_conditioning", False)
        ) and state_dim > 0
        if self.use_state_conditioning:
            self.state_projector = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, int(action_model_cfg.action_hidden_dim)),
            )
        else:
            self.state_projector = None

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        # self.hidden_dim = config.framework.action_model.action_hidden_dim

        robot_type = getattr(
            getattr(getattr(self.config, "datasets", None), "vla_data", None),
            "robot_type",
            None,
        )
        self.robot_type = robot_type
        action_dim = int(self.config.framework.action_model.action_dim)
        self._unit_quaternion_slices: Tuple[Tuple[int, int], ...] = (
            ((3, 7), (21, 25))
            if robot_type == "EgoS2_Adamu" and action_dim == 36
            else ()
        )

        self.action_token = "🔍"
        action_token_ids = self.qwen_vl_interface.processor.tokenizer(
            self.action_token,
            add_special_tokens=False,
        )["input_ids"]
        if not action_token_ids:
            raise RuntimeError(
                f"Action marker {self.action_token!r} produced no tokenizer ids for "
                f"{self.qwen_vl_interface.__class__.__name__}."
            )
        # Some tokenizers encode the marker as one id (Qwen2.5/Qwen3), while
        # others encode it as several ids (Qwen3.5).  Keep the complete
        # sequence; extracting only the first id silently creates the wrong
        # number/position of action queries.
        self.action_token_ids = tuple(int(token_id) for token_id in action_token_ids)
        # Keep the singular attribute for callers that inspect it.  The
        # gather helper below accepts either an int or a full id sequence.
        self.action_token_id = (
            self.action_token_ids[0]
            if len(self.action_token_ids) == 1
            else self.action_token_ids
        )

        # L1 loss
        # Keep the unreduced loss so the trainer can identify the physical
        # dataset sample behind a spike.  The scalar mean remains the training
        # objective used by previous checkpoints.
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly regress future actions (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 7]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        # Optionally prepend discretised proprioceptive state tokens to each instruction (π₀.5 style).
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        # step 0: add special action token to instruction
        action_tokens = (
            self.action_token * self.chunk_len
        )  # can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_ids
            )  # [B, chunk_len, H]
            action_queries = self._add_continuous_state_condition(action_queries, state)
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)
            pred_actions = self._project_unit_quaternions(
                pred_actions, self._unit_quaternion_slices
            )

            # Label alignment: take the last chunk_len segment
            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)
            actions_target = self._project_unit_quaternions(
                actions_target, self._unit_quaternion_slices
            )

            # Compute L1 loss only over real action positions.  The dataset
            # intentionally keeps true pause/hold actions; the mask excludes
            # only synthetic positions past the episode boundary.
            action_valid_mask = build_action_valid_mask(
                examples,
                horizon=self.action_horizon,
                device=pred_actions.device,
                dtype=pred_actions.dtype,
            )
            dimension_weights = None
            action_loss_weights = self.config.trainer.get("action_loss_weights", {}) or {}
            if self.robot_type == "EgoS2_Adamu" and pred_actions.shape[-1] == 36:
                position_weight = float(action_loss_weights.get("eef_position", 1.0))
                quaternion_weight = float(action_loss_weights.get("eef_quaternion", 1.0))
                hand_weight = float(action_loss_weights.get("hand", 1.0))
                dimension_weights = torch.full(
                    (36,), hand_weight, device=pred_actions.device, dtype=pred_actions.dtype
                )
                dimension_weights[0:3] = position_weight
                dimension_weights[3:7] = quaternion_weight
                dimension_weights[18:21] = position_weight
                dimension_weights[21:25] = quaternion_weight
            action_loss_per_sample = compute_masked_action_l1_loss(
                pred_actions,
                actions_target,
                action_valid_mask,
                dimension_weights=dimension_weights,
            )
            action_loss = action_loss_per_sample.mean()

        return {
            "action_loss": action_loss,
            "action_loss_per_sample": action_loss_per_sample,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        # Optionally prepend discretised proprioceptive state tokens to each instruction (π₀.5 style).
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # step 0: add special action token to instruction
        action_tokens = (
            self.action_token * self.chunk_len
        )  # can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_ids
            )  # [B, chunk_len, H]
            action_queries = self._add_continuous_state_condition(action_queries, state)
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)
            pred_actions = self._project_unit_quaternions(
                pred_actions, self._unit_quaternion_slices
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @staticmethod
    def _project_unit_quaternions(
        actions: torch.Tensor,
        slices: Tuple[Tuple[int, int], ...],
    ) -> torch.Tensor:
        """Project configured quaternion action slices onto canonical unit quaternions.

        Quaternion components are one geometric object: normalizing each component
        independently destroys the rotation representation.  We therefore apply a
        joint L2 normalization, use identity for near-zero predictions, and choose
        the canonical hemisphere so equivalent ``q``/``-q`` labels agree.
        """
        if not slices:
            return actions

        projected = actions
        for start, end in slices:
            q = projected[..., start:end]
            eps = torch.finfo(q.dtype).eps
            norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)

            q = q / norm.clamp_min(eps)

            identity = torch.zeros_like(q)
            identity[..., 0] = 1.0
            q = torch.where(norm > eps, q, identity)

            sign = torch.where(
                q[..., :1] < 0,
                -torch.ones_like(q[..., :1]),
                torch.ones_like(q[..., :1]),
            )
            q = q * sign
            projected = torch.cat(
                (projected[..., :start], q, projected[..., end:]), dim=-1
            )

        return projected

    def _add_continuous_state_condition(
        self,
        action_queries: torch.Tensor,
        state: List[np.ndarray] | None,
    ) -> torch.Tensor:
        """Add the optional continuous proprioceptive state embedding.

        The dataloader packs one state observation as ``[1, state_dim]`` per
        example.  Only the observation at the current frame is used; the
        action horizon remains represented by the separate causal marker
        queries.  Returning the original tensor when disabled keeps the
        legacy QwenOFT graph and checkpoint behavior unchanged.
        """
        if not self.use_state_conditioning or self.state_projector is None or state is None:
            return action_queries

        state_tensor = torch.as_tensor(
            np.asarray(state),
            device=action_queries.device,
            dtype=torch.float32,
        )
        if state_tensor.ndim == 3 and state_tensor.shape[1] == 1:
            state_tensor = state_tensor[:, 0, :]
        elif state_tensor.ndim != 2:
            raise ValueError(
                "Expected state shaped [B, 1, state_dim] or [B, state_dim], got "
                f"{tuple(state_tensor.shape)}"
            )

        projected = self.state_projector(state_tensor)
        if projected.shape[0] != action_queries.shape[0] or projected.shape[-1] != action_queries.shape[-1]:
            raise ValueError(
                "Continuous state projection shape does not match action queries: "
                f"projection={tuple(projected.shape)}, queries={tuple(action_queries.shape)}"
            )
        return action_queries + projected.to(dtype=action_queries.dtype).unsqueeze(1)

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,  # [B, L, H]
        input_ids: torch.Tensor,  # [B, L]
        action_token_id=None,  # Can be int or List[int]
    ) -> torch.Tensor:
        """
        Extract one hidden state per complete action marker.

        The marker may be encoded as one tokenizer id or as a sequence of
        ids.  Matching the complete sequence is important: matching only the
        first id can accidentally select a sub-token and makes the query
        count depend on the tokenizer.  For a multi-id marker, use the final
        marker position so its hidden state has seen the complete marker.
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int or List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id must not be None")

        if input_ids is None:
            raise RuntimeError("QwenOFT requires input_ids to locate action markers.")
        if last_hidden.ndim != 3 or input_ids.ndim != 2:
            raise ValueError(
                "Expected last_hidden [B, L, H] and input_ids [B, L], got "
                f"{tuple(last_hidden.shape)} and {tuple(input_ids.shape)}"
            )

        device = input_ids.device
        batch_size, seq_len, hidden_size = last_hidden.shape
        if input_ids.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                "last_hidden and input_ids must have the same [B, L], got "
                f"{tuple(last_hidden.shape[:2])} and {tuple(input_ids.shape)}"
            )

        if isinstance(action_token_id, torch.Tensor):
            marker_ids = [int(token_id) for token_id in action_token_id.reshape(-1).tolist()]
        elif isinstance(action_token_id, set):
            marker_ids = [int(token_id) for token_id in sorted(action_token_id)]
        elif isinstance(action_token_id, (list, tuple)):
            marker_ids = [int(token_id) for token_id in action_token_id]
        else:
            marker_ids = [int(action_token_id)]
        if not marker_ids:
            raise ValueError("action_token_id must contain at least one tokenizer id")

        marker = torch.tensor(marker_ids, device=device, dtype=input_ids.dtype)
        marker_len = len(marker_ids)
        if marker_len == 1:
            marker_starts = input_ids.eq(marker[0]).nonzero(as_tuple=False)
            matches_by_row = [
                marker_starts[marker_starts[:, 0] == row, 1]
                for row in range(batch_size)
            ]
        else:
            if seq_len < marker_len:
                matches_by_row = [torch.empty(0, device=device, dtype=torch.long)] * batch_size
            else:
                windows = input_ids.unfold(1, marker_len, 1)
                match_mask = windows.eq(marker.view(1, 1, -1)).all(dim=-1)
                matches_by_row = [
                    match_mask[row].nonzero(as_tuple=False).flatten()
                    for row in range(batch_size)
                ]

        selected_positions = []
        for row, starts in enumerate(matches_by_row):
            # Match spans must not overlap.  This also makes the behavior
            # deterministic if a future tokenizer emits a self-overlapping
            # marker sequence.
            selected_starts = []
            next_available_start = 0
            for start in starts.tolist():
                if start >= next_available_start:
                    selected_starts.append(start)
                    next_available_start = start + marker_len

            if len(selected_starts) < self.chunk_len:
                raise RuntimeError(
                    "Insufficient complete action markers: "
                    f"sample={row}, required={self.chunk_len}, "
                    f"found={len(selected_starts)}, marker_ids={marker_ids}"
                )

            # Keep the final chunk_len markers, matching the previous
            # behavior when an instruction contains an incidental marker.
            selected_starts = selected_starts[-self.chunk_len :]
            # Use the final id of each marker as the query position.
            selected_positions.append(
                torch.tensor(
                    [start + marker_len - 1 for start in selected_starts],
                    device=device,
                    dtype=torch.long,
                )
            )

        selected_pos = torch.stack(selected_positions, dim=0)
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, hidden_size)
        return last_hidden.gather(dim=1, index=expanded_index)

    # Discretised state → instruction prefix (π₀.5 style); shared with QwenPI_v3.
    add_discretized_state_to_instruction = staticmethod(add_discretized_state_to_instruction)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"[train] Action Loss (with state): {action_loss.item()}")

    predict_output = model.predict_action(examples=[batch[0]])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[infer] Predicted Action shape: {normalized_actions.shape}")

    # Backward-compat: examples without `state` should still work.
    sample_no_state = {k: v for k, v in sample.items() if k != "state"}
    forward_no_state = model([sample_no_state, sample_no_state])
    print(f"[train] Action Loss (no state): {forward_no_state['action_loss'].item()}")
    predict_no_state = model.predict_action(examples=[sample_no_state])
    print(f"[infer] Predicted Action shape (no state): {predict_no_state['normalized_actions'].shape}")

    print("Finished")
