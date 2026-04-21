"""Persistent PromptEncoder — keeps Gemma 12B in VRAM between calls.

Stock PromptEncoder loads Gemma from disk on every __call__ then frees it.
On H100 this wastes ~10s per inference.

This module patches _text_encoder_ctx to cache the built Gemma model in
VRAM and reuse it across calls.

If bitsandbytes is available, Gemma is quantized to int8 (~12GB instead of
~24GB). This frees ~12GB of VRAM headroom so GFPGAN enhance can safely run
alongside the persistent Gemma without OOM.
"""

import logging
import threading
from contextlib import contextmanager

import torch

from ltx_pipelines.utils.blocks import PromptEncoder

logger = logging.getLogger(__name__)

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
    logger.info("bitsandbytes available — Gemma will be quantized to int8")
except Exception:
    _BNB_AVAILABLE = False
    logger.info("bitsandbytes not available — Gemma loaded in bf16")


def _to_int8_inplace(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Recursively replace nn.Linear layers with bitsandbytes Linear8bitLt (int8).

    Pattern: set weight from CPU data (frees bf16 GPU copy), then move layer to GPU
    so bitsandbytes stores the int8 weights on device (half the VRAM of bf16).
    """
    for name, module in list(model.named_children()):
        if isinstance(module, torch.nn.Linear):
            q = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            # .cpu() frees the bf16 GPU copy before assigning int8 params
            q.weight = bnb.nn.Int8Params(
                module.weight.data.cpu(),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if module.bias is not None:
                q.bias = module.bias
            # Move int8 weights to GPU (bitsandbytes stores as int8, not bf16)
            q = q.to(device)
            setattr(model, name, q)
            del module
        else:
            _to_int8_inplace(module, device)
    return model


class PersistentPromptEncoder(PromptEncoder):
    """PromptEncoder that caches Gemma in VRAM between calls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cached_text_encoder = None
        self._warmup_lock = threading.Lock()

    def warmup(self) -> None:
        """Build Gemma and cache it in VRAM. Thread-safe."""
        with self._warmup_lock:
            if self._cached_text_encoder is not None:
                return
            logger.info("PersistentPromptEncoder: loading Gemma into VRAM...")
            enc = (
                self._text_encoder_builder
                .build(device=self._device, dtype=self._dtype)
                .eval()
            )
            if _BNB_AVAILABLE:
                logger.info("PersistentPromptEncoder: quantizing Gemma to int8...")
                vram_before = torch.cuda.memory_allocated() / 1e9
                _to_int8_inplace(enc, self._device)
                torch.cuda.empty_cache()
                vram_after = torch.cuda.memory_allocated() / 1e9
                logger.info(
                    f"PersistentPromptEncoder: int8 quantization done "
                    f"({vram_before:.1f}GB -> {vram_after:.1f}GB)"
                )
            self._cached_text_encoder = enc
            logger.info("PersistentPromptEncoder: Gemma cached in VRAM")

    @contextmanager
    def _text_encoder_ctx(self, streaming_prefetch_count=None, **kwargs):
        """Yield Gemma for text encoding, then clear bitsandbytes CxB cache to free ~15GB."""
        with self._warmup_lock:
            if self._cached_text_encoder is None:
                # Background reload hasn't finished — build synchronously now
                logger.info("PersistentPromptEncoder: Gemma not ready, loading synchronously...")
                self._cached_text_encoder = (
                    self._text_encoder_builder
                    .build(device=self._device, dtype=self._dtype)
                    .eval()
                )
        yield self._cached_text_encoder

        # After text encoding, free VRAM so spatial upsampler has enough contiguous memory.
        if _BNB_AVAILABLE:
            # int8: clear CxB transformed-weight cache (~15GB), keeps int8 weights (~8GB)
            self._clear_bnb_forward_cache()
        else:
            # bf16: no CxB cache, but clear PyTorch allocator fragmentation
            torch.cuda.empty_cache()

    def _clear_bnb_forward_cache(self) -> None:
        """Delete CxB/SB from each Linear8bitLt layer to free the transformed-weight cache."""
        if self._cached_text_encoder is None:
            return
        freed = 0
        for module in self._cached_text_encoder.modules():
            if isinstance(module, bnb.nn.Linear8bitLt):
                state = module.state
                if getattr(state, "CxB", None) is not None:
                    del state.CxB
                    state.CxB = None
                    freed += 1
                if getattr(state, "SB", None) is not None:
                    del state.SB
                    state.SB = None
        if freed:
            torch.cuda.empty_cache()
            logger.debug(f"PersistentPromptEncoder: cleared CxB for {freed} int8 layers")

    def free(self) -> None:
        """Explicitly release Gemma from VRAM."""
        if self._cached_text_encoder is not None:
            self._cached_text_encoder.to("meta")
            self._cached_text_encoder = None
            torch.cuda.empty_cache()
            logger.info("PersistentPromptEncoder: Gemma freed")
