"""Train a model if needed and serve it.

    python scripts/serve.py            # trains in-memory and serves on :5000

Then:
    curl -s localhost:5000/health
    curl -s -X POST localhost:5000/predict -H 'content-type: application/json' \\
      -d '{"age":64,"bmi":31.2,"num_chronic":2,"exercise_days":1,"sex":"M","region":"south","smoker":"yes"}'
"""

from __future__ import annotations

from insurance_risk import create_app, make_dataset, train


def main() -> None:
    pipe, metrics = train(make_dataset(n=4000, seed=0))
    print("trained:", metrics)
    create_app(pipe).run(host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
