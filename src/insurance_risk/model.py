"""Training pipeline: preprocessing + classifier, as one sklearn Pipeline.

Keeping preprocessing inside the pipeline matters for serving: the exact same
scaling and one-hot encoding that were fit at training time travel with the
model, so there is no train/serve skew to get wrong by hand.
"""

from __future__ import annotations

from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def train(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "accuracy": round(float(accuracy_score(y_te, proba >= 0.5)), 4),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }
    return pipe, metrics


def save(pipe: Pipeline, path: str) -> None:
    joblib.dump(pipe, path)


def load(path: str) -> Pipeline:
    return joblib.load(path)
