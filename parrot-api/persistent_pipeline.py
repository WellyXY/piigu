"""Persistent pipeline — keeps the transformer in VRAM between calls.

The stock DistilledPipeline loads the 43GB transformer from disk on EVERY
__call__ (twice: Stage 1 + Stage 2), then frees it. On H100 this wastes
~28s per inference.

This module patches DiffusionStage so the transformer is built once, then
ALL LoRAs (nsfw, motion, position) are hot-swapped in-place via tensor
add/subtract (~2s) instead of rebuilding from disk (~28s).
"""

import logging
import sys
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


def _dbg(msg: str) -> None:
    """Force-print to stderr so it appears in Ray worker logs regardless of logging config."""
    print(f"[PIPELINE] {msg}", file=sys.stderr, flush=True)


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
    raw_sd = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            raw_sd[k] = f.get_tensor(k)

    op = LTXV_LORA_COMFY_RENAMING_MAP
    renamed = {op.apply_to_key(k): v for k, v in raw_sd.items()}

    suffix_index: dict[str, str] = {}
    for pname in param_names:
        if not pname.endswith(".weight"):
            continue
        suffix_index[pname] = pname
        dot = pname.find(".")
        if dot != -1:
            suffix_index[pname[dot + 1:]] = pname

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

    total_magnitude = sum(d.abs().sum().item() for d in deltas.values()) if deltas else 0.0
    _dbg(f"_compute_lora_deltas: {lora_path.split('/')[-1]} matched={matched}/{len(lora_prefixes)} strength={strength:.3f} magnitude={total_magnitude:.4f}")
    logger.info(f"_compute_lora_deltas: matched {matched}/{len(lora_prefixes)} LoRA layers from {lora_path}")
    return deltas


class PersistentDiffusionStage(DiffusionStage):
    """DiffusionStage with fully hot-swappable LoRAs.

    At warmup:
      - Builds transformer with base LoRAs (nsfw, motion) fused at default weights.
      - Pre-computes delta tensors for all POSITION LoRAs on CPU (small files, fast).
      - Applies the initial position LoRA delta.
      - Base LoRA (nsfw, motion) deltas are NOT pre-computed (avoids ~100GB RAM spike).

    On weight change:
      - Position swap: subtract old delta, add new delta from pre-computed cache. ~2s.
      - Base swap: load LoRA file on-demand, compute + apply net delta, discard. ~3-5s.
    """

    def __init__(self, *args, **kwargs) -> None:
        # position_loras: {pos_key -> (path, strength)}
        self._position_loras: dict[tuple, tuple[str, float]] = kwargs.pop("position_loras", {})
        self._initial_position_key: tuple | None = kwargs.pop("initial_position_key", None)
        # scalable_loras: {name -> (path, default_strength)} — nsfw, motion
        self._scalable_loras_config: dict[str, tuple[str, float]] = kwargs.pop("scalable_loras", {})
        super().__init__(*args, **kwargs)
        self._cached_transformer: X0Model | None = None
        self._param_dict: dict[str, torch.nn.Parameter] | None = None
        # Position LoRA deltas (CPU): {pos_name -> {param_name -> delta at strength=1.0}}
        # Strength is NOT baked in — applied at swap time so any weight works.
        self._position_deltas: dict[str, dict[str, torch.Tensor]] = {}
        self._current_position_key: tuple | None = None   # primary (pos_name, strength)
        self._current_secondary_key: tuple | None = None  # secondary stacked LoRA, or None
        # Base LoRA weights — no pre-computed deltas stored; applied on-demand to save RAM
        self._current_base_weights: dict[str, float] = {}

    def warmup(self) -> None:
        """Build transformer with base LoRAs fused, pre-compute position deltas for hot-swap."""
        if self._cached_transformer is not None:
            return

        _dbg("warmup: building transformer (nsfw+motion fused)...")
        logger.info("PersistentDiffusionStage: building transformer (nsfw+motion fused)...")
        self._cached_transformer = self._build_transformer()
        _dbg("warmup: transformer cached in VRAM")
        logger.info("PersistentDiffusionStage: transformer cached in VRAM")

        self._param_dict = dict(self._cached_transformer.named_parameters())
        param_names = set(self._param_dict.keys())
        dtype = self._dtype

        # Log first few param names for debugging
        sample_params = [p for p in param_names if ".weight" in p][:5]
        _dbg(f"warmup: sample param names (first 5): {sample_params}")

        # Record base LoRA default weights (already fused into transformer at build time).
        # No delta pre-compute — we apply net deltas on-demand to avoid ~100GB RAM spike.
        for name, (lora_path, default_w) in self._scalable_loras_config.items():
            self._current_base_weights[name] = default_w
            _dbg(f"warmup: base LoRA '{name}' fused at weight={default_w}")
            logger.info(f"PersistentDiffusionStage: base LoRA '{name}' fused at weight={default_w}")

        # Pre-compute position LoRA deltas at strength=1.0 (weight applied at swap time)
        seen_positions: set[str] = set()
        for pos_key, (lora_path, _strength) in self._position_loras.items():
            pos_name = pos_key[0]
            if pos_name in seen_positions:
                continue
            seen_positions.add(pos_name)
            _dbg(f"warmup: pre-computing position LoRA deltas for '{pos_name}'...")
            logger.info(f"PersistentDiffusionStage: pre-computing position LoRA deltas for '{pos_name}'...")
            self._position_deltas[pos_name] = _compute_lora_deltas(
                lora_path, 1.0, param_names, torch.device("cpu"), dtype  # always 1.0
            )
        if self._position_deltas:
            _dbg(f"warmup: {len(self._position_deltas)} position LoRA deltas loaded: {list(self._position_deltas.keys())}")
            logger.info(f"PersistentDiffusionStage: {len(self._position_deltas)} position LoRA deltas loaded")

        # Apply initial position LoRA
        if self._initial_position_key is not None:
            pos_name = self._initial_position_key[0]
            if pos_name in self._position_deltas:
                _dbg(f"warmup: applying initial position LoRA '{pos_name}'")
                self._apply_position_delta(self._initial_position_key)

    # -- Base LoRA (nsfw, motion) hot-swap — on-demand to avoid RAM OOM at warmup --

    def swap_base_loras(self, weights: dict[str, float]) -> None:
        """Rescale base LoRAs (nsfw, motion) per-request.

        Applies the NET delta (new_w - current_w) layer-by-layer on GPU:
        load A+B for one layer → compute B@A on GPU → apply → free.
        Peak VRAM overhead: one layer's A+B (~few MB). ~2-5s vs ~40-80s on CPU.
        """
        device = self._device
        dtype = self._dtype

        for name, new_w in weights.items():
            if name not in self._scalable_loras_config:
                continue
            current_w = self._current_base_weights.get(name, 0.0)
            if abs(new_w - current_w) < 1e-4:
                continue

            lora_path, _ = self._scalable_loras_config[name]
            net_strength = new_w - current_w
            _dbg(f"swap_base_loras: rescaling '{name}' {current_w:.3f} -> {new_w:.3f} (net={net_strength:+.3f}, GPU layer-by-layer)")
            logger.info(
                f"PersistentDiffusionStage: rescaling '{name}' {current_w:.3f} -> {new_w:.3f} "
                f"(net={net_strength:+.3f}, GPU layer-by-layer)"
            )

            op = LTXV_LORA_COMFY_RENAMING_MAP
            param_names = set(self._param_dict.keys())

            # Build suffix index once
            suffix_index: dict[str, str] = {}
            for pname in param_names:
                if not pname.endswith(".weight"):
                    continue
                suffix_index[pname] = pname
                dot = pname.find(".")
                if dot != -1:
                    suffix_index[pname[dot + 1:]] = pname

            matched = 0
            total_magnitude = 0.0
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                all_keys = set(f.keys())
                # Build renamed_prefix → orig_prefix map by renaming the actual lora_A key.
                # This mirrors _compute_lora_deltas: rename ALL keys first, derive prefix from
                # renamed key — NOT from orig key. Applying op to a synthetic ".weight" key
                # is wrong because the rename map matches the full ".lora_A.weight" pattern.
                prefix_map: dict[str, str] = {}  # renamed_prefix → orig_prefix
                for k in all_keys:
                    if k.endswith(".lora_A.weight"):
                        orig_pfx = k[: -len(".lora_A.weight")]
                        renamed_key = op.apply_to_key(k)
                        renamed_pfx = renamed_key[: -len(".lora_A.weight")]
                        prefix_map[renamed_pfx] = orig_pfx

                for renamed_prefix, orig_prefix in prefix_map.items():
                    if f"{orig_prefix}.lora_B.weight" not in all_keys:
                        continue
                    # Look up transformer param via renamed prefix (matches suffix_index built
                    # from named_parameters which already have the model's native naming)
                    lora_weight_key = f"{renamed_prefix}.weight"
                    param_name = suffix_index.get(lora_weight_key)
                    if param_name is None:
                        continue
                    # Load A, B → GPU, compute delta in-place, apply, free immediately
                    A = f.get_tensor(f"{orig_prefix}.lora_A.weight").to(device=device, dtype=dtype)
                    B = f.get_tensor(f"{orig_prefix}.lora_B.weight").to(device=device, dtype=dtype)
                    delta = (B @ A) * net_strength
                    del A, B
                    total_magnitude += delta.abs().sum().item()
                    self._param_dict[param_name].data.add_(delta)
                    del delta
                    matched += 1

            torch.cuda.empty_cache()
            self._current_base_weights[name] = new_w
            _dbg(f"swap_base_loras: '{name}' rescaled to {new_w:.3f} (matched {matched}/{len(prefix_map)} layers, magnitude={total_magnitude:.4f})")
            logger.info(f"PersistentDiffusionStage: '{name}' rescaled to {new_w:.3f} (matched {matched} layers)")

    # -- Position LoRA hot-swap --

    def _apply_position_delta(self, pos_key: tuple, update_key: bool = True) -> None:
        pos_name, strength = pos_key
        deltas = self._position_deltas[pos_name]
        device = self._device
        total_mag = 0.0
        for param_name, delta in deltas.items():
            scaled = delta.to(device) * strength
            total_mag += scaled.abs().sum().item()
            self._param_dict[param_name].data.add_(scaled)
        if update_key:
            self._current_position_key = pos_key
        _dbg(f"_apply_position_delta: applied '{pos_name}' strength={strength:.3f}, layers={len(deltas)}, magnitude={total_mag:.4f}")

    def _remove_position_delta(self, pos_key: tuple) -> None:
        pos_name, strength = pos_key
        deltas = self._position_deltas[pos_name]
        device = self._device
        total_mag = 0.0
        for param_name, delta in deltas.items():
            scaled = delta.to(device) * strength
            total_mag += scaled.abs().sum().item()
            self._param_dict[param_name].data.sub_(scaled)
        _dbg(f"_remove_position_delta: removed '{pos_name}' strength={strength:.3f}, layers={len(deltas)}, magnitude={total_mag:.4f}")

    def swap_position_lora(self, new_pos_key: tuple) -> None:
        """Hot-swap the position LoRA. Any weight supported — delta stored at 1.0, scaled at apply."""
        if new_pos_key == self._current_position_key:
            _dbg(f"swap_position_lora: already at {new_pos_key}, skip")
            return

        new_pos_name = new_pos_key[0]
        _dbg(f"swap_position_lora: {self._current_position_key} -> {new_pos_key}")
        logger.info(f"PersistentDiffusionStage: hot-swap {self._current_position_key} -> {new_pos_key}")

        if self._current_position_key is not None:
            cur_pos_name = self._current_position_key[0]
            if cur_pos_name in self._position_deltas:
                self._remove_position_delta(self._current_position_key)

        if new_pos_name in self._position_deltas:
            self._apply_position_delta(new_pos_key)
        else:
            _dbg(f"swap_position_lora: WARNING no delta for '{new_pos_name}', skipping apply")
            logger.warning(f"No delta found for position '{new_pos_name}', skipping")
            self._current_position_key = new_pos_key

        _dbg(f"swap_position_lora: complete, current_position_key={self._current_position_key}")
        logger.info("PersistentDiffusionStage: hot-swap complete")

    def ensure_loras(self, primary_key: tuple, secondary_key: tuple | None = None) -> None:
        """Apply primary + optional secondary position LoRA, swapping cleanly as needed."""
        # Step 1: remove stale secondary if it changed
        if self._current_secondary_key != secondary_key:
            if self._current_secondary_key is not None:
                sec_name = self._current_secondary_key[0]
                if sec_name in self._position_deltas:
                    self._remove_position_delta(self._current_secondary_key)
            self._current_secondary_key = None

        # Step 2: swap primary (existing logic handles skip-if-same)
        self.swap_position_lora(primary_key)

        # Step 3: apply new secondary if needed
        if secondary_key is not None and self._current_secondary_key != secondary_key:
            sec_name = secondary_key[0]
            if sec_name in self._position_deltas:
                self._apply_position_delta(secondary_key, update_key=False)
                self._current_secondary_key = secondary_key
                _dbg(f"ensure_loras: secondary '{sec_name}' applied at {secondary_key[1]:.3f}")
            else:
                _dbg(f"ensure_loras: WARNING no delta for secondary '{sec_name}'")

    def _transformer_ctx(self, streaming_prefetch_count=None, **kwargs):
        if self._cached_transformer is None:
            self.warmup()
        return _noop_ctx(self._cached_transformer)

    def free(self) -> None:
        if self._cached_transformer is not None:
            torch.cuda.synchronize()
            self._cached_transformer.to("meta")
            self._cached_transformer = None
            self._param_dict = None
            self._position_deltas.clear()
            self._current_position_key = None
            self._current_secondary_key = None
            self._current_base_weights.clear()
            cleanup_memory()
            _dbg("free: transformer freed")
            logger.info("PersistentDiffusionStage: transformer freed")

    @property
    def is_loaded(self) -> bool:
        return self._cached_transformer is not None
