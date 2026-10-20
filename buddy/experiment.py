"""
Runs every strategy on the same simulated readers and measures regret.

Regret for one session = (the reader's best possible success probability)
- (success probability of the settings the strategy chose). Zero means the
strategy picked that reader's best settings. Using probabilities rather than
the random 0/1 outcome measures the choice, not the luck.

Readers arrive one after another and each reads for `sessions` sessions,
so later readers can benefit from what a strategy learned earlier.
"""
from dataclasses import dataclass

import numpy as np

from buddy.readers import sample_reader


@dataclass
class RunResult:
    strategy: str
    regret: np.ndarray        # (readers, sessions)
    chosen_prob: np.ndarray   # (readers, sessions) success probability of the chosen settings
    best_prob: np.ndarray     # (readers,)


def run(strategy_cls, n_readers: int, sessions: int, seed: int) -> RunResult:
    reader_rng = np.random.default_rng(seed)           # same readers for every strategy
    outcome_rng = np.random.default_rng(seed + 10_000)  # same coin flips for every strategy
    strategy = strategy_cls(np.random.default_rng(seed + 20_000))
    readers = [sample_reader(reader_rng) for _ in range(n_readers)]

    regret = np.zeros((n_readers, sessions))
    chosen = np.zeros((n_readers, sessions))
    for r, reader in enumerate(readers):
        for s in range(sessions):
            arm = strategy.choose(r, reader.context)
            reward = reader.read(arm, outcome_rng)
            strategy.update(r, reader.context, arm, reward)
            chosen[r, s] = reader.success_probs[arm]
            regret[r, s] = reader.best_prob - reader.success_probs[arm]
    return RunResult(strategy_cls.name, regret, chosen, np.array([rd.best_prob for rd in readers]))


def summarize(results: list[RunResult], near_best: float = 0.02) -> dict:
    """Averages over seeds for one strategy."""
    regret = np.stack([r.regret for r in results])        # (seeds, readers, sessions)
    chosen = np.stack([r.chosen_prob for r in results])
    best = np.stack([r.best_prob for r in results])[..., None]
    sessions = regret.shape[-1]
    late = slice(sessions - 5, sessions)

    cumulative = regret.sum(axis=2).mean(axis=1)           # per seed: mean cumulative regret per reader
    return {
        "strategy": results[0].strategy,
        "cumulative_regret": float(cumulative.mean()),
        "cumulative_regret_sd": float(cumulative.std()),
        "first5_success": float(chosen[..., :5].mean()),
        "last5_success": float(chosen[..., late].mean()),
        "last5_near_best": float((chosen[..., late] >= best - near_best).mean()),
        "best_possible_success": float(best.mean()),
        "regret_by_session": regret.mean(axis=(0, 1)),          # learning within a reader
        "first5_regret_by_reader": regret[..., :5].mean(axis=(0, 2)),  # cold start as more readers arrive
    }
