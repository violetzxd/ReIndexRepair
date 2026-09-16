from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .core import RepairAction


@dataclass(frozen=True)
class CandidateScore:
    action: RepairAction
    edit_tokens: int
    attack_risk: float
    utility_loss: float
    fidelity: str

    def feasible(self, epsilon: float, delta: float) -> bool:
        return self.attack_risk <= epsilon and self.utility_loss <= delta


def pareto_frontier(candidates: Sequence[CandidateScore]) -> list[CandidateScore]:
    frontier: list[CandidateScore] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            no_worse = (
                other.edit_tokens <= candidate.edit_tokens
                and other.attack_risk <= candidate.attack_risk
                and other.utility_loss <= candidate.utility_loss
            )
            strictly_better = (
                other.edit_tokens < candidate.edit_tokens
                or other.attack_risk < candidate.attack_risk
                or other.utility_loss < candidate.utility_loss
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item.edit_tokens, item.attack_risk, item.utility_loss))


def exhaustive_minimum(
    actions: Sequence[RepairAction],
    evaluate_f2: Callable[[RepairAction], CandidateScore],
    *,
    epsilon: float,
    delta: float,
) -> CandidateScore | None:
    """Certified minimum only for an explicitly enumerated finite repair grammar."""
    evaluated = [evaluate_f2(action) for action in actions]
    feasible = [score for score in evaluated if score.feasible(epsilon, delta)]
    if not feasible:
        return None
    return min(feasible, key=lambda item: (item.edit_tokens, item.utility_loss, item.attack_risk))

