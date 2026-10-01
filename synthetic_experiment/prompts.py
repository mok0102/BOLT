"""Shared serialization contract for SFT, DPO, sampling, and likelihood."""

from __future__ import annotations

import json
import re
from decimal import Decimal, ROUND_HALF_UP

SYSTEM_PROMPT = (
    "You optimize a two-dimensional Branin function. Return only one point "
    "as [x1, x2], with x1 in [-5, 10] and x2 in [0, 15]."
)


def format_float(value: float) -> str:
    """Format a prompt number to exactly two decimals using schoolbook rounding."""

    return str(Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def format_task_t(value: float) -> str:
    """Format the task context to exactly three decimal places."""

    return str(Decimal(str(float(value))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def task_context(task_t: float) -> str:
    return f"task_t={format_task_t(task_t)}\nPropose a good initial point for this task."


def point_text(point) -> str:
    return f"[{format_float(point[0])}, {format_float(point[1])}]"


def parse_point(text: str) -> list[float] | None:
    match = re.search(r"\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]", text)
    if not match:
        # DPO checkpoints can occasionally serialize the two coordinates as
        # adjacent singleton lists, e.g. "[9.46]\n[2.57]". Treat that as the
        # same point while retaining the usual bounds validation below.
        match = re.search(
            r"\[\s*([-+0-9.eE]+)\s*\]\s*\[\s*([-+0-9.eE]+)\s*\]",
            text,
        )
    if not match:
        return None
    try:
        point = [float(match.group(1)), float(match.group(2))]
    except ValueError:
        return None
    if -5.0 <= point[0] <= 10.0 and 0.0 <= point[1] <= 15.0:
        return point
    return None


def messages(task_t: float, point=None) -> list[dict[str, str]]:
    result = [{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": task_context(task_t)}]
    if point is not None:
        result.append({"role": "assistant", "content": point_text(point)})
    return result
