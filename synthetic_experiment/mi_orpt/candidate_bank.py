from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..prompts import point_text


@dataclass(frozen=True)
class EligibleCandidate:
    seq: str
    x: tuple[float, float]
    y: float


def build_eligible_bank(path: Path) -> list[EligibleCandidate]:
    unique = {}
    for row in pd.read_csv(path).itertuples():
        try:
            point = ast.literal_eval(row.train_x)
            x = (float(point[0]), float(point[1]))
            y = float(row.train_y)
        except (ValueError, TypeError, SyntaxError, IndexError):
            continue
        if len(point) != 2 or not all(map(math.isfinite, (*x, y))):
            continue
        seq = point_text(x)
        unique.setdefault(seq, EligibleCandidate(seq, x, y))
    return list(unique.values())


def min_bank_size_needed(m: int, num_reserved: int = 2) -> int:
    return m - 1 + num_reserved
