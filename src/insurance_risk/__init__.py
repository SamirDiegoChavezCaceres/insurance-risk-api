"""Health-risk classification with an sklearn pipeline, served over a Flask API."""

from .api import create_app
from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET, make_dataset
from .model import build_pipeline, load, save, train

__all__ = [
    "make_dataset",
    "FEATURES",
    "NUMERIC",
    "CATEGORICAL",
    "TARGET",
    "build_pipeline",
    "train",
    "save",
    "load",
    "create_app",
]
