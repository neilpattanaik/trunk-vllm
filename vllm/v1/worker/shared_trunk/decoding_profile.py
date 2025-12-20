from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodingProfile:
    """Lightweight per-request decoding controls."""

    eos_logit_bias: float = 0.0
    max_tokens_cap: int | None = None
    min_tokens_floor: int | None = None
    stop_strings: list[str] | None = None
