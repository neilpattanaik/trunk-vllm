from __future__ import annotations

from typing import Dict, List

import torch


class BoundaryStore:
    """Per-request store for boundary activations H_B."""

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self._store: Dict[str, List[torch.Tensor]] = {}

    def init_for_request(self, req_id: str) -> None:
        self._store[req_id] = []

    def append(self, req_id: str, hb_chunk: torch.Tensor) -> None:
        if hb_chunk.dim() == 3 and hb_chunk.size(0) == 1:
            hb_chunk = hb_chunk[0]
        if hb_chunk.dim() != 2:
            raise ValueError(f"Expected hb_chunk dim=2, got {hb_chunk.dim()}.")
        hb_chunk = hb_chunk.detach()
        if hb_chunk.device != self.device or hb_chunk.dtype != self.dtype:
            hb_chunk = hb_chunk.to(device=self.device, dtype=self.dtype)
        # Make a local copy to avoid aliasing if the source buffer is mutated.
        hb_chunk = hb_chunk.clone()
        self._store.setdefault(req_id, []).append(hb_chunk)

    def get_prefix(self, req_id: str, total_len: int) -> torch.Tensor:
        if req_id not in self._store:
            raise KeyError(f"Missing boundary store for req_id={req_id}")
        chunks = self._store[req_id]
        if not chunks:
            raise RuntimeError(f"No boundary chunks for req_id={req_id}")
        hb = torch.cat(chunks, dim=0)
        if hb.size(0) < total_len:
            raise RuntimeError(
                f"Insufficient H_B cached tokens: {hb.size(0)} < {total_len}"
            )
        return hb[:total_len].contiguous()

    def num_tokens(self, req_id: str) -> int:
        if req_id not in self._store:
            raise KeyError(f"Missing boundary store for req_id={req_id}")
        return sum(chunk.size(0) for chunk in self._store[req_id])

    def free(self, req_id: str) -> None:
        self._store.pop(req_id, None)
