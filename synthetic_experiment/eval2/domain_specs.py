"""Constants for synthetic Branin figures."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainSpec:
    name: str = 'Synthetic Branin'


SYNTHETIC = DomainSpec()
