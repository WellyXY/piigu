"""Persistent pipeline — keeps the transformer in VRAM between calls.

The stock DistilledPipeline loads the 43GB transformer from disk on EVERY
__call__ (twice: Stage 1 + Stage 2), then frees it. On H100 this wastes
~28s per inference.

This module patches DiffusionStage so the transformer is built once with
nsfw + motion LoRAs fused in, then position LoRAs are hot-swapped in-place
(tensor add/subtract, ~2s) instead of rebuilding from disk (~28s).
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from safetensors import safe_open

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.model.transformer import X0Model
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.helpers import cleanup_memory

logger = logging.getLogger(__name__)


@contextmanager
def _noop_ctx(model: X0Model) -> Iterator[X0Model]:
    """Yield the model without freeing it on exit."""
    yield model


def _compute_lora_deltas(
    lora_path: str,
    strength: float,
    param_names: set[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Load a LoRA file and compute merged delta tensors keyed by transformer param name.

    For each layer: delta = B @ A * strength
    Keys are stripped of the 'diffusion_model.' prefix to match transformer param names.
    """
    # Load raw LoRA state dict
    raw_sd = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            raw_sd[k] = f.get_tensor(k)

    # Apply SDOps key renaming (strips 'diffusion_model.' prefix)
    op = LTXV_LORA_COMFY_RENAMING_MAP
    renamed = {op.apply_to_key(k): v for k, v in raw_sd.items()}

    # Compute delta for each transformer weight that has a matching LoRA
    deltas = {}
    for param_name in param_names:
        if not param_name.endswith(".weight"):
            continue
        prefix = param_name[: -len(".weight")]
        key_a = f"{prefix}.lora_A.weight"
        key_b = f"{prefix}.lora_B.weight"
        if key_a not in renamed or key_b not in renamed:
            continue
        A = renamed[key_a].to(device=device, dtype=dtype)
        B = renamed[key_b].to(device=device, dtype=dtype)
        deltas[param_name] = (B @ A) * strength
        del A, B

    return deltas


class PersistentDiffusionStage(DiffusionStage):
    """DiffusionStage that caches the transformer in VRAM with hot-swappable position LoRA.

    At warmup:
      - Builds transformer with nsfw + motion LoRAs fused (constant).
      - Pre-computes delta tensors for ALL position LoRAs and caches them in VRAM.
      - Applies the initial position LoRA delta.

    On position change (swap_position_lora):
      - Subtracts old position LoRA delta from transformer weights.
      - Adds new position LoRA delta.
      - ~2s instead of ~28s rebuild.
    """

    def __init__(self, *args, **kwargs) -> None:
        # position_loras: dict of {position_key -> (path, strength)} for hot-swap
        self._position_loras: dict[tuple, tuple[str, float]] = kwargs.pop("position_loras", {})
        self._initial_position_key: tuple | None = kwargs.pop("initial_position_key", None)
        super().__init__(*args, **kwargs)
        self._cached_transformer: X0Model | None = None
        # Cached transformer param dict for fast lookup during swap
        self._param_dict: dict[str, torch.nn.Parameter] | None = None
        # Pre-computed position LoRA delta tensors: {position_key -> {param_name -> delta}}
        self._position_deltas: dict[tuple, dict[str, torch.Tensor]] = {}
        self._current_position_key: tuple | None = None

    def warmup(self) -> None:
        """Build transformer with base LoRAs fused, pre-compute all position LoRA deltas."""
        if self._cached_transformer is not None:
            return

        logger.info("PersistentDiffusionStage: building transformer (nsfw+motion LoRAs fused)...")
        self._cached_transformer = self._build_transformer()
        logger.info("PersistentDiffusionStage: transformer cached in VRAM")

        # Cache param dict for fast hot-swap
        self._param_dict = dict(self._cached_transformer.named_parameters())
        param_names = set(self._param_dict.keys())

        # Pre-compute delta tensors for all position LoRAs
        device = self._device
        dtype = self._dtype
        for pos_key, (lora_path, strength) in self._position_loras.items():
            logger.info(f"PersistentDiffusionStage: pre-computing LoRA deltas for {pos_key}...")
            self._position_deltas[pos_key] = _compute_lora_deltas(
                lora_path, strength, param_names, device, dtype
            )
        if self._position_loras:
            logger.info(f"PersistentDiffusionStage: {len(self._position_loras)} position LoRA deltas loaded")

        # Apply initial position LoRA
        if self._initial_position_key is not None and self._initial_position_key in self._position_deltas:
            self._apply_position_delta(self._initial_position_key)

    def _apply_position_delta(self, pos_key: tuple) -> None:
        """Add position LoRA delta to transformer weights in-place."""
        deltas = self._position_deltas[pos_key]
        for param_name, delta in deltas.items():
            self._param_dict[param_name].data.add_(delta)
        self._current_position_key = pos_key

    def _remove_position_delta(self, pos_key: tuple) -> None:
        """Subtract position LoRA delta from transformer weights in-place."""
        deltas = self._position_deltas[pos_key]
        for param_name, delta in deltas.items():
            self._param_dict[param_name].data.sub_(delta)

    def swap_position_lora(self, new_pos_key: tuple) -> None:
        """Hot-swap the position LoRA. ~2s vs ~28s rebuild."""
        if new_pos_key == self._current_position_key:
            return

        logger.info(f"PersistentDiffusionStage: hot-swap {self._current_position_key} -> {new_pos_key}")

        # Remove old position LoRA
        if self._current_position_key is not None and self._current_position_key in self._position_deltas:
            self._remove_position_delta(self._current_position_key)

        # Apply new position LoRA
        if new_pos_key in self._position_deltas:
            self._apply_position_delta(new_pos_key)
        else:
            logger.warning(f"No delta found for position key {new_pos_key}, skipping")
            self._current_position_key = new_pos_key

        logger.info("PersistentDiffusionStage: hot-swap complete")

    def _transformer_ctx(self, streaming_prefetch_count=None, **kwargs):
        if self._cached_transformer is None:
            self.warmup()
        return _noop_ctx(self._cached_transformer)

    def free(self) -> None:
        """Explicitly release the cached transformer and all LoRA deltas."""
        if self._cached_transformer is not None:
            torch.cuda.synchronize()
            self._cached_transformer.to("meta")
            self._cached_transformer = None
            self._param_dict = None
            self._position_deltas.clear()
            self._current_position_key = None
            cleanup_memory()
            logger.info("PersistentDiffusionStage: transformer freed")

    @property
    def is_loaded(self) -> bool:
        return self._cached_transformer is not None
