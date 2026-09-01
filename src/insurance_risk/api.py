"""A Flask API that serves the trained pipeline.

    GET  /health   -> liveness
    POST /predict   {feature: value, ...} -> {risk_probability, risk_label}

The model is injected into ``create_app`` so the app is trivially testable with
Flask's test client, and input is validated so a missing field is a clean 400
rather than a 500 from deep inside sklearn.
"""

from __future__ import annotations

import pandas as pd
from flask import Flask, jsonify, request

from .data import FEATURES


def create_app(model) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "body must be a JSON object"}), 400
        missing = [f for f in FEATURES if f not in payload]
        if missing:
            return jsonify({"error": "missing fields", "fields": missing}), 400

        row = pd.DataFrame([{f: payload[f] for f in FEATURES}])
        proba = float(model.predict_proba(row)[0, 1])
        return {"risk_probability": round(proba, 4), "risk_label": int(proba >= 0.5)}

    return app
