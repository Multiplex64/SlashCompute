"""Duty-cycle GPU usage: work for ``t``, then sleep so busy/wall ~= percent/100."""

from __future__ import annotations

import asyncio


def sleep_s(busy_s: float, percent: int) -> float:
    """How long to idle after ``busy_s`` seconds of compute."""
    p = max(int(percent), 0) / 100.0
    if p <= 0:
        return 3600.0
    if p >= 1.0:
        return 0.0
    return max(busy_s, 0.0) * (1.0 - p) / p


def make_pace(percent: int):
    async def pace(busy_s: float) -> None:
        wait = sleep_s(busy_s, percent)
        if wait > 0:
            await asyncio.sleep(wait)

    return pace
