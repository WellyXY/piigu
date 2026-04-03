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

    Matches LoRA layers to transformer params by suffix, so it works regardless of
    whatever prefix X0Model uses in named_parameters() (e.g. 'model.', none, etc.).
    """
    # Load raw LoRA state dict
    raw_sd = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            raw_sd[k] = f.get_tensor(k)

    # Apply SDOps key renaming (strips 'diffusion_model.' prefix from LoRA keys)
    op = LTXV_LORA_COMFY_RENAMING_MAP
    renamed = {op.apply_to_key(k): v for k, v in raw_sd.items()}

    # Build suffix -> param_name index for O(1) suffix lookup.
    # Handles cases where X0Model wraps the transformer in a submodule (adds a prefix).
    # Maps every possible suffix of each param name to that param name.
    # e.g. 'model.transformer_blocks.0.attn.to_k.weight'
    #   -> {'model.transformer_blocks.0.attn.to_k.weight': ...,
    #       'transformer_blocks.0.attn.to_k.weight': ...,
    #       ...}
    # We only need one level of stripping in practice (one wrapper module prefix).
    suffix_index: dict[str, str] = {}
    for pname in param_names:
        if not pname.endswith(".weight"):
            continue
        suffix_index[pname] = pname  # exact
        # Strip one prefix segment
        dot = pname.find(".")
        if dot != -1:
            suffix_index[pname[dot + 1:]] = pname

    # Collect LoRA base prefixes from keys ending in .lora_A.weight
    lora_prefixes = [k[: -len(".lora_A.weight")] for k in renamed if k.endswith(".lora_A.weight")]

    deltas = {}
    matched = 0
    for lora_prefix in lora_prefixes:
        key_a = f"{lora_prefix}.lora_A.weight"
        key_b = f"{lora_prefix}.lora_B.weight"
        if key_b not in renamed:
            continue

        lora_weight_key = f"{lora_prefix}.weight"
        param_name = suffix_index.get(lora_weight_key)
        if param_name is None:
            continue

        A = renamed[key_a].to(device=device, dtype=dtype)
        B = renamed[key_b].to(device=device, dtype=dtype)
        deltas[param_name] = (B @ A) * strength
        del A, B
        matched += 1

    logger.info(f"_compute_lora_deltas: matched {matched}/{len(lora_prefixes)} LoRA layers from {lora_path}")
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

        # Pre-compute delta tensors for all position LoRAs.
        # Deltas are computed and stored on CPU to avoid VRAM OOM (transformer already ~43GB).
        # They are moved to GPU only during apply/remove (add_ accepts cross-device tensors via .to()).
        dtype = self._dtype
        for pos_key, (lora_path, strength) in self._position_loras.items():
            logger.info(f"PersistentDiffusionStage: pre-computing LoRA deltas for {pos_key}...")
            self._position_deltas[pos_key] = _compute_lora_deltas(
                lora_path, strength, param_names, torch.device("cpu"), dtype
            )
        if self._position_loras:
            logger.info(f"PersistentDiffusionStage: {len(self._position_loras)} position LoRA deltas loaded")

        # Apply initial position LoRA
        if self._initial_position_key is not None and self._initial_position_key in self._position_deltas:
            self._apply_position_delta(self._initial_position_key)

    def _apply_position_delta(self, pos_key: tuple) -> None:
        """Add position LoRA delta to transformer weights in-place.
        Deltas are stored on CPU; moved to GPU temporarily for each add_."""
        deltas = self._position_deltas[pos_key]
        device = self._device
        for param_name, delta in deltas.items():
            self._param_dict[param_name].data.add_(delta.to(device))
        self._current_position_key = pos_key

    def _remove_position_delta(self, pos_key: tuple) -> None:
        """Subtract position LoRA delta from transformer weights in-place.
        Deltas are stored on CPU; moved to GPU temporarily for each sub_."""
        deltas = self._position_deltas[pos_key]
        device = self._device
        for param_name, delta in deltas.items():
            self._param_dict[param_name].data.sub_(delta.to(device))

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
