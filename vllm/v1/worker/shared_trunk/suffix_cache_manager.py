from __future__ import annotations

from typing import Any, Callable, Dict

import torch

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.utils import extract_layer_index
from vllm.attention.backends.abstract import AttentionMetadata


class SuffixCacheManager:
    """Manages two KV caches for suffix layers and switching between them."""

    def __init__(self, boundary_layer: int):
        self.boundary_layer = boundary_layer
        self._attn_layers: Dict[str, AttentionLayerBase] = {}
        self._kv_cache_a: Dict[str, torch.Tensor] = {}
        self._kv_cache_b: Dict[str, torch.Tensor] = {}
        self._active_suffix_id: int = 0

    def initialize_from_kv_caches(
        self,
        vllm_config: VllmConfig,
        kv_caches: Dict[str, torch.Tensor],
    ) -> None:
        self._attn_layers.clear()
        self._kv_cache_a.clear()
        self._kv_cache_b.clear()

        attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
        for layer_name, attn_module in attn_layers.items():
            try:
                layer_idx = extract_layer_index(layer_name)
            except AssertionError:
                continue
            if layer_idx < self.boundary_layer:
                continue
            kv_cache = kv_caches.get(layer_name)
            if kv_cache is None:
                continue
            self._attn_layers[layer_name] = attn_module
            self._kv_cache_a[layer_name] = kv_cache
            self._kv_cache_b[layer_name] = torch.empty_like(kv_cache)

        if not self._attn_layers:
            raise RuntimeError(
                "No suffix attention layers found for debug suffix switch."
            )

        self._active_suffix_id = 0
        self.set_active_suffix(0)

    def set_active_suffix(self, suffix_id: int) -> None:
        if suffix_id not in (0, 1):
            raise ValueError(f"Unknown suffix id: {suffix_id}")
        if suffix_id == self._active_suffix_id:
            return
        kv_map = self._kv_cache_a if suffix_id == 0 else self._kv_cache_b
        for layer_name, kv_cache in kv_map.items():
            attn_module = self._attn_layers[layer_name]
            if not attn_module.kv_cache:
                attn_module.kv_cache = [kv_cache]
            else:
                attn_module.kv_cache[0] = kv_cache
        self._active_suffix_id = suffix_id

    def reset(self) -> None:
        self.set_active_suffix(0)

    def rebuild_from_boundary(
        self,
        *,
        model: Any,
        hb: torch.Tensor,
        boundary_layer: int,
        prompt_len: int,
        target_len: int,
        device: torch.device,
        dtype: torch.dtype,
        vllm_config: VllmConfig,
        build_prefill_attn_metadata: Callable[[int], dict[str, AttentionMetadata]],
        build_decode_attn_metadata: Callable[[int], dict[str, AttentionMetadata]],
    ) -> None:
        if prompt_len < 0 or target_len < 0 or prompt_len > target_len:
            raise ValueError("Invalid prompt_len/target_len for suffix rebuild.")
        if hb.size(0) < target_len:
            raise RuntimeError(
                f"Insufficient H_B cached tokens: {hb.size(0)} < {target_len}"
            )
        if target_len == 0:
            return
        if hb.device != device or hb.dtype != dtype:
            hb = hb.to(device=device, dtype=dtype)

        if prompt_len > 0:
            hb_prompt = hb[:prompt_len]
            positions_prompt = torch.arange(
                prompt_len, device=device, dtype=torch.int64
            )
            attn_metadata = build_prefill_attn_metadata(prompt_len)
            with set_forward_context(
                attn_metadata,
                vllm_config,
                num_tokens=prompt_len,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                model.forward_suffix_prefill_from_hb(
                    hb_prompt,
                    positions_prompt,
                    boundary_layer,
                )

        for offset in range(target_len - prompt_len):
            position = prompt_len + offset
            hb_token = hb[position : position + 1]
            positions = torch.tensor([position], device=device, dtype=torch.int64)
            attn_metadata = build_decode_attn_metadata(position)
            with set_forward_context(
                attn_metadata,
                vllm_config,
                num_tokens=1,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                model.forward_suffix_prefill_from_hb(
                    hb_token,
                    positions,
                    boundary_layer,
                )
