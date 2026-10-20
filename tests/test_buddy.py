import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from buddy.experiment import run
from buddy.readers import Reader, sample_reader
from buddy.settings import ARMS, DEFAULT_ARM, N_ARMS
from buddy.strategies import ALL_STRATEGIES, FixedDefault, HierarchicalThompson, LinearThompson, ThompsonPerReader


def test_24_distinct_arms_and_default_is_standard():
    assert N_ARMS == 24 == len(set(ARMS))
    assert ARMS[DEFAULT_ARM].font == "standard" and ARMS[DEFAULT_ARM].speed == 1.0


def test_readers_have_valid_probabilities_and_differ():
    rng = np.random.default_rng(0)
    readers = [sample_reader(rng) for _ in range(200)]
    for r in readers:
        assert ((r.success_probs > 0) & (r.success_probs < 1)).all()
    # personal quirks: readers do not all share one best setting
    assert len({r.best_arm for r in readers}) > 5


def test_same_seed_same_readers_for_every_strategy():
    a = run(FixedDefault, n_readers=5, sessions=3, seed=7)
    b = run(LinearThompson, n_readers=5, sessions=3, seed=7)
    assert np.allclose(a.best_prob, b.best_prob)


def test_regret_is_never_negative():
    for cls in ALL_STRATEGIES:
        assert (run(cls, n_readers=10, sessions=10, seed=1).regret >= -1e-12).all(), cls.name


def test_thompson_finds_an_obvious_best_arm():
    """One reader for whom a single setting is far better: Thompson sampling must settle on it."""
    probs = np.full(N_ARMS, 0.2)
    probs[5] = 0.9
    reader = Reader(context=np.array([1.0, 0, 0, 0]), success_probs=probs)
    strategy, rng = ThompsonPerReader(np.random.default_rng(0)), np.random.default_rng(1)
    picks = []
    for _ in range(300):
        arm = strategy.choose(0, reader.context)
        strategy.update(0, reader.context, arm, reader.read(arm, rng))
        picks.append(arm)
    assert np.mean(np.array(picks[-100:]) == 5) > 0.8


def test_contextual_strategies_beat_fixed_default():
    default = run(FixedDefault, n_readers=100, sessions=10, seed=3).regret.sum()
    for cls in (LinearThompson, HierarchicalThompson):
        assert run(cls, n_readers=100, sessions=10, seed=3).regret.sum() < 0.8 * default, cls.name
