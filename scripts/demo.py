"""End-to-end walkthrough: train a model, serve it, call it, validate input.

    python scripts/demo.py

Uses the Flask test client, so no server or port is needed. Synthetic data.
"""

from __future__ import annotations

from insurance_risk import create_app, make_dataset, train

HIGH_RISK = {"age": 64, "bmi": 34.0, "num_chronic": 3, "exercise_days": 0,
             "sex": "M", "region": "south", "smoker": "yes"}
LOW_RISK = {"age": 28, "bmi": 22.0, "num_chronic": 0, "exercise_days": 5,
            "sex": "F", "region": "north", "smoker": "no"}


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    rule("1. Train on synthetic data")
    pipe, metrics = train(make_dataset(n=4000, seed=0))
    print(f"  metrics: {metrics}")

    rule("2. Serve it (in-process Flask test client)")
    client = create_app(pipe).test_client()
    print(f"  GET /health -> {client.get('/health').get_json()}")

    rule("3. Score two profiles")
    for label, profile in [("high-risk", HIGH_RISK), ("low-risk", LOW_RISK)]:
        body = client.post("/predict", json=profile).get_json()
        print(f"  {label:9} -> risk_probability={body['risk_probability']:<6} "
              f"risk_label={body['risk_label']}")

    rule("4. A missing field is a clean 400, not a 500")
    bad = {k: v for k, v in HIGH_RISK.items() if k != "smoker"}
    res = client.post("/predict", json=bad)
    print(f"  status={res.status_code}  body={res.get_json()}")


if __name__ == "__main__":
    main()
