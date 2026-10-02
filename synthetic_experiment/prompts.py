"""Shared serialization contract for SFT, DPO, sampling, and likelihood.

One task descriptor formatter (`task_descriptor`), used by every training/
inference call site (`messages()` here, plus OptFormer's and LLAMBO's own
history-conditioned prompts prepend this same line) -- previously each of
those 3 reimplemented an inline `f"task_t=..."` line independently.
Following the peptide domain's own prompt convention (embed the task
descriptor directly as the user-turn payload, not a growing key=value
line).
"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

from .branin import BRANIN_BOUNDS, BraninTaskTransform

SYSTEM_PROMPT = (
    "You optimize a two-dimensional Branin function. Each task is the "
    "canonical Branin landscape rotated and scaled about the box center, "
    "then shifted, as described in the user message. Return only one point "
    f"as [x1, x2], with x1 in [{BRANIN_BOUNDS[0][0]:g}, {BRANIN_BOUNDS[0][1]:g}] "
    f"and x2 in [{BRANIN_BOUNDS[1][0]:g}, {BRANIN_BOUNDS[1][1]:g}]."
)


def format_float(value: float) -> str:
    """Format a prompt number to exactly two decimals using schoolbook rounding."""

    return str(Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


_DESCRIPTOR_TEMPLATE = (
    "Branin task: rotated {rotation_deg:.1f} deg and scaled by {scale:.3f} "
    "about the box center, then shifted by ({shift0:+.2f}, {shift1:+.2f})."
)
_DESCRIPTOR_RE = re.compile(
    r"Branin task: rotated ([-+0-9.]+) deg and scaled by ([0-9.]+) about the box center, "
    r"then shifted by \(([-+0-9.]+),\s*([-+0-9.]+)\)\."
)


def task_descriptor(transform: BraninTaskTransform) -> str:
    """The one line of text that tells the model which task it's solving.
    Quantized to the same precision as the sampler rounds to (branin.py's
    BraninTaskTransform itself doesn't round; task_splits.py's builder
    does), so this is an exact serialization of the task, not a lossy
    display rounding of it."""
    return _DESCRIPTOR_TEMPLATE.format(
        rotation_deg=transform.rotation_deg, scale=transform.scale,
        shift0=transform.shift[0], shift1=transform.shift[1],
    )


def parse_task_descriptor(text: str) -> BraninTaskTransform | None:
    """Exact inverse of `task_descriptor`, for a round-trip self-check."""
    match = _DESCRIPTOR_RE.search(text)
    if not match:
        return None
    rotation_deg, scale, shift0, shift1 = match.groups()
    try:
        return BraninTaskTransform(
            shift=(float(shift0), float(shift1)), rotation_deg=float(rotation_deg), scale=float(scale),
        )
    except ValueError:
        return None


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
    if BRANIN_BOUNDS[0][0] <= point[0] <= BRANIN_BOUNDS[0][1] and BRANIN_BOUNDS[1][0] <= point[1] <= BRANIN_BOUNDS[1][1]:
        return point
    return None


def messages(transform: BraninTaskTransform, point=None) -> list[dict[str, str]]:
    result = [{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": task_descriptor(transform)}]
    if point is not None:
        result.append({"role": "assistant", "content": point_text(point)})
    return result
