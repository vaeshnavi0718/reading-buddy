"""
Reading settings the buddy can choose between: the bandit's "arms".

Four settings, combined into 2 x 2 x 2 x 3 = 24 arms:
- font:     standard, or a dyslexia-friendly font (e.g. Lexend)
- spacing:  normal, or wide letter/line spacing
- chunking: the full page at once, or short chunks of a few lines
- speed:    read-aloud at 0.8x, 1.0x or 1.2x
"""
from dataclasses import dataclass
from itertools import product

import numpy as np

FONTS = ("standard", "dyslexia_friendly")
SPACINGS = ("normal", "wide")
CHUNKINGS = ("full_page", "short_chunks")
SPEEDS = (0.8, 1.0, 1.2)


@dataclass(frozen=True)
class Setting:
    font: str
    spacing: str
    chunking: str
    speed: float

    def features(self) -> np.ndarray:
        """[bias, dyslexia font, wide spacing, short chunks, slow speech, fast speech]"""
        return np.array([
            1.0,
            float(self.font == "dyslexia_friendly"),
            float(self.spacing == "wide"),
            float(self.chunking == "short_chunks"),
            float(self.speed < 1.0),
            float(self.speed > 1.0),
        ])

    def label(self) -> str:
        return f"{self.font}/{self.spacing}/{self.chunking}/{self.speed}x"


ARMS: tuple[Setting, ...] = tuple(Setting(*combo) for combo in product(FONTS, SPACINGS, CHUNKINGS, SPEEDS))
ARM_FEATURES = np.stack([arm.features() for arm in ARMS])  # (24, 6)
N_ARMS, ARM_DIM = ARM_FEATURES.shape

# What a typical reading app shows everyone: the one-size-fits-all baseline.
DEFAULT_ARM = ARMS.index(Setting("standard", "normal", "full_page", 1.0))
