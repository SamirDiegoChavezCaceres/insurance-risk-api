# insurance-risk-api

[![CI](https://github.com/SamirDiegoChavezCaceres/insurance-risk-api/actions/workflows/ci.yml/badge.svg)](https://github.com/SamirDiegoChavezCaceres/insurance-risk-api/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Train a health-risk classifier and **serve it over a REST API** - the full loop
from data to a running prediction endpoint, not just a notebook.

Built on synthetic data, so it runs end to end with nothing to download.

## What it shows

- **One sklearn `Pipeline`** carries preprocessing (scaling + one-hot encoding)
  together with the classifier, so the exact transforms fit at training time are
  reused at serving time - no train/serve skew.
- **A Flask API** (`/predict`, `/health`) with input validation: a missing
  feature is a clean `400`, not a `500` from inside sklearn.
- **Testable serving**: the model is injected into `create_app`, so the API is
  tested with Flask's test client, no live server needed.

## Run it

```bash
pip install -e .
python scripts/serve.py
```

```bash
curl -s localhost:5000/health
curl -s -X POST localhost:5000/predict -H 'content-type: application/json' \
  -d '{"age":64,"bmi":31.2,"num_chronic":2,"exercise_days":1,"sex":"M","region":"south","smoker":"yes"}'
# {"risk_probability": 0.83, "risk_label": 1}
```

Train and persist a model separately:

```bash
python scripts/train.py model.joblib
```

## Results

On the synthetic data the model reaches **ROC-AUC ~0.77** on a held-out split
(fixed seed, so the number is reproducible). Reproduce:

```bash
python scripts/train.py
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers that the model learns the synthetic signal (ROC-AUC above a baseline),
and the API contract: a valid request returns a probability, a missing field is
a 400, and `/health` is live.

## Limitations and next steps

- Trained on synthetic data, so the metrics show the pipeline works, not
  real-world accuracy.
- The API has no auth, rate limiting, or model versioning yet.
- Next: add request logging, a model registry, and schema validation on the
  input (e.g. pydantic).

## License

MIT.
