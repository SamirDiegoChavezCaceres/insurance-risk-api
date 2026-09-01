"""A synthetic health-risk dataset.

No real or proprietary data is used. Rows are generated from a known logistic
relationship so a model has a real (but noisy) signal to learn, and the demo /
tests run with nothing to download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["age", "bmi", "num_chronic", "exercise_days"]
CATEGORICAL = ["sex", "region", "smoker"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "high_risk"


def make_dataset(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    bmi = rng.normal(27, 5, n).clip(15, 50)
    num_chronic = rng.poisson(0.7, n).clip(0, 5)
    exercise_days = rng.integers(0, 7, n)
    sex = rng.choice(["F", "M"], n)
    region = rng.choice(["north", "south", "east", "west"], n)
    smoker = rng.choice(["yes", "no"], n, p=[0.25, 0.75])

    logit = (
        -4.0
        + 0.04 * age
        + 0.08 * (bmi - 25)
        + 0.9 * num_chronic
        - 0.15 * exercise_days
        + 1.3 * (smoker == "yes")
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    high_risk = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "bmi": bmi.round(1),
            "num_chronic": num_chronic,
            "exercise_days": exercise_days,
            "sex": sex,
            "region": region,
            "smoker": smoker,
            TARGET: high_risk,
        }
    )
