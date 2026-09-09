"""Use unique completed plans; censored runtimes are not exact utilities."""
import ast
import math
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class EligibleCandidate:
    seq: str
    y: float

def build_eligible_bank(path):
    bank = {}
    for row in pd.read_csv(path).itertuples():
        if int(row.censoring) != 0 or not math.isfinite(row.train_y) or row.train_y >= 0:
            continue
        try:
            plan = ast.literal_eval(row.train_x)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(plan, list) or not plan or len(plan) % 5 or any(type(x) is not int for x in plan):
            continue
        seq = str(plan)
        bank.setdefault(seq, EligibleCandidate(seq, float(row.train_y)))
    return list(bank.values())

def min_bank_size_needed(m, num_reserved=2):
    return m - 1 + num_reserved
