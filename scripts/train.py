"""Train on synthetic data and save the model.

    python scripts/train.py [model_path]
"""

from __future__ import annotations

import sys

from insurance_risk import make_dataset, save, train


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "model.joblib"
    pipe, metrics = train(make_dataset(n=4000, seed=0))
    save(pipe, path)
    print(f"saved {path}")
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
