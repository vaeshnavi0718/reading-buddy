"""
Strategies for choosing reading settings, simplest first.

Every strategy answers two calls:
  choose(reader_id, context) -> arm      which settings to show this session
  update(reader_id, context, arm, reward) what happened (1 = passed the check)

Per-reader strategies learn each reader from scratch. The two Thompson
sampling variants at the end are contextual: they also learn from other
readers, which helps when a new reader arrives with no history.
"""
from collections import defaultdict

import numpy as np

from buddy.readers import CONTEXT_DIM
from buddy.settings import ARM_DIM, ARM_FEATURES, DEFAULT_ARM, N_ARMS


class FixedDefault:
    """What most apps do: the same settings for everyone, no learning."""
    name = "Fixed default"

    def __init__(self, rng): pass
    def choose(self, reader_id, context): return DEFAULT_ARM
    def update(self, reader_id, context, arm, reward): pass


class RandomChoice:
    """Pure exploration: a random setting every session."""
    name = "Random"

    def __init__(self, rng): self.rng = rng
    def choose(self, reader_id, context): return int(self.rng.integers(N_ARMS))
    def update(self, reader_id, context, arm, reward): pass


class _PerReaderCounts:
    """Success/attempt counts per reader and arm."""

    def __init__(self):
        self.successes = defaultdict(lambda: np.zeros(N_ARMS))
        self.attempts = defaultdict(lambda: np.zeros(N_ARMS))

    def update(self, reader_id, context, arm, reward):
        self.attempts[reader_id][arm] += 1
        self.successes[reader_id][arm] += reward


class EpsilonGreedy(_PerReaderCounts):
    """Usually pick the best-so-far setting; with probability epsilon, try a random one."""
    name = "Epsilon-greedy (per reader)"

    def __init__(self, rng, epsilon: float = 0.1):
        super().__init__()
        self.rng, self.epsilon = rng, epsilon

    def choose(self, reader_id, context):
        attempts, successes = self.attempts[reader_id], self.successes[reader_id]
        if self.rng.random() < self.epsilon or attempts.sum() == 0:
            return int(self.rng.integers(N_ARMS))
        means = np.divide(successes, attempts, out=np.full(N_ARMS, 0.5), where=attempts > 0)
        return int(self.rng.choice(np.flatnonzero(means == means.max())))


class UCB1(_PerReaderCounts):
    """Optimism: pick the setting whose success rate could plausibly be highest."""
    name = "UCB1 (per reader)"

    def __init__(self, rng): super().__init__()

    def choose(self, reader_id, context):
        attempts, successes = self.attempts[reader_id], self.successes[reader_id]
        untried = np.flatnonzero(attempts == 0)
        if untried.size:
            return int(untried[0])
        total = attempts.sum()
        scores = successes / attempts + np.sqrt(2 * np.log(total) / attempts)
        return int(np.argmax(scores))


class ThompsonPerReader(_PerReaderCounts):
    """Keep a Beta belief about each setting's success rate; sample from it and pick the best sample."""
    name = "Thompson sampling (per reader)"

    def __init__(self, rng): super().__init__(); self.rng = rng

    def choose(self, reader_id, context):
        attempts, successes = self.attempts[reader_id], self.successes[reader_id]
        samples = self.rng.beta(1 + successes, 1 + attempts - successes)
        return int(np.argmax(samples))


def _interaction_features(context: np.ndarray) -> np.ndarray:
    """Every (setting feature x context feature) pair, for each arm: shape (N_ARMS, ARM_DIM * CONTEXT_DIM)."""
    return np.einsum("ad,c->adc", ARM_FEATURES, context).reshape(N_ARMS, ARM_DIM * CONTEXT_DIM)


class LinearThompson:
    """
    Contextual Thompson sampling shared by all readers.

    A Bayesian linear model of reward from (setting x context) features,
    learned across every reader. New readers benefit immediately from what
    worked for similar readers, but personal quirks are never learned.
    """
    name = "Contextual Thompson (shared)"

    def __init__(self, rng, prior_precision: float = 1.0, noise_sd: float = 0.5):
        self.rng, self.noise_var = rng, noise_sd ** 2
        dim = ARM_DIM * CONTEXT_DIM
        self.precision = prior_precision * np.eye(dim)
        self.b = np.zeros(dim)

    def posterior(self):
        cov = np.linalg.inv(self.precision)
        return cov @ self.b, cov

    def choose(self, reader_id, context):
        mean, cov = self.posterior()
        theta = self.rng.multivariate_normal(mean, self.noise_var * cov, method="cholesky")
        return int(np.argmax(_interaction_features(context) @ theta))

    def update(self, reader_id, context, arm, reward):
        x = _interaction_features(context)[arm]
        self.precision += np.outer(x, x)
        self.b += reward * x


class HierarchicalThompson(LinearThompson):
    """
    Population model + personal model.

    The shared contextual model (above) predicts what should work for a
    reader like this one; that prediction is the starting belief for a small
    per-reader model, which then personalizes as the reader's own sessions
    come in. Cold start from the population, personal fit from the reader.

    personal_sd (how far a reader may differ from the population, in
    success-probability units) was tuned on held-out seeds 100-104; reported
    results use seeds 0-19.
    """
    name = "Hierarchical Thompson (population + personal)"

    def __init__(self, rng, prior_precision: float = 1.0, noise_sd: float = 0.5, personal_sd: float = 0.05):
        super().__init__(rng, prior_precision, noise_sd)
        self.personal_prior_precision = 1.0 / personal_sd ** 2
        self.reader_xx = defaultdict(lambda: np.zeros((ARM_DIM, ARM_DIM)))
        self.reader_xr = defaultdict(lambda: np.zeros(ARM_DIM))

    def choose(self, reader_id, context):
        # Sample the population model (not its mean): Thompson sampling explores
        # by sampling, and using the mean here made the population model greedy.
        global_mean, global_cov = self.posterior()
        global_theta = self.rng.multivariate_normal(global_mean, self.noise_var * global_cov, method="cholesky")
        # Population prediction for this context, expressed as per-setting weights.
        prior_mean = global_theta.reshape(ARM_DIM, CONTEXT_DIM) @ context
        precision = self.personal_prior_precision * np.eye(ARM_DIM) + self.reader_xx[reader_id] / self.noise_var
        cov = np.linalg.inv(precision)
        mean = cov @ (self.personal_prior_precision * prior_mean + self.reader_xr[reader_id] / self.noise_var)
        theta = self.rng.multivariate_normal(mean, cov, method="cholesky")
        return int(np.argmax(ARM_FEATURES @ theta))

    def update(self, reader_id, context, arm, reward):
        super().update(reader_id, context, arm, reward)
        a = ARM_FEATURES[arm]
        self.reader_xx[reader_id] += np.outer(a, a)
        self.reader_xr[reader_id] += reward * a


ALL_STRATEGIES = (FixedDefault, RandomChoice, EpsilonGreedy, UCB1, ThompsonPerReader, LinearThompson, HierarchicalThompson)
