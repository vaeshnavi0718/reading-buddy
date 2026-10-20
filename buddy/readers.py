"""
Simulated readers.

There are no real users yet, so the bandits are evaluated on a population of
simulated readers. Each reader has a context the app can see and hidden
preferences it can't. A session's reward is whether the reader passes a
short comprehension check (1) or not (0).

ASSUMPTIONS (chosen to be plausible, not measured; results depend on them):
- Context the app can see: decoding difficulty (0/1), confident reader
  (0/1), reading on a phone (0/1).
- Readers with decoding difficulty do better with a dyslexia-friendly font,
  wide spacing and slower speech, and worse with fast speech.
- Less confident readers do better with short chunks; confident readers
  prefer full pages and faster speech.
- Short chunks help on a phone; wide spacing costs a little there (more
  scrolling).
- On top of these population patterns, every reader has personal quirks
  (random per-reader deviations), so the population pattern alone never
  finds everyone's best setting. That's what personalization has to learn.
"""
from dataclasses import dataclass

import numpy as np

from buddy.settings import ARM_DIM, ARM_FEATURES

CONTEXT_DIM = 4  # [bias, decoding difficulty, confident reader, on phone]

# Population effect of each setting feature (rows) per context feature (cols).
# Rows: bias, dyslexia font, wide spacing, short chunks, slow speech, fast speech
# Cols: bias, difficulty, confident, phone
POPULATION_EFFECTS = np.array([
    [-0.2, -0.5,  0.6,  0.0],   # baseline success (logit)
    [ 0.0,  0.6,  0.0,  0.0],   # dyslexia-friendly font helps with decoding difficulty
    [ 0.0,  0.5,  0.0, -0.2],   # wide spacing helps difficulty, small cost on phones
    [ 0.1,  0.0, -0.5,  0.3],   # short chunks help less confident readers and phones
    [ 0.0,  0.4, -0.4,  0.0],   # slow speech helps difficulty, bores confident readers
    [-0.1, -0.6,  0.3,  0.0],   # fast speech hurts difficulty, suits confident readers
])
PERSONAL_SD = 0.4  # spread of each reader's personal deviation from the population


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Reader:
    context: np.ndarray        # what the app sees
    success_probs: np.ndarray  # hidden: P(pass comprehension check) for each arm

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.success_probs))

    @property
    def best_prob(self) -> float:
        return float(self.success_probs.max())

    def read(self, arm: int, rng: np.random.Generator) -> int:
        """One session with the given settings: 1 if the comprehension check is passed."""
        return int(rng.random() < self.success_probs[arm])


def sample_reader(rng: np.random.Generator) -> Reader:
    difficulty = rng.random() < 0.5
    confident = rng.random() < 0.4
    phone = rng.random() < 0.5
    context = np.array([1.0, difficulty, confident, phone], dtype=float)

    weights = POPULATION_EFFECTS @ context + rng.normal(0.0, PERSONAL_SD, size=ARM_DIM)
    probs = _sigmoid(ARM_FEATURES @ weights)
    return Reader(context=context, success_probs=probs)
