from insurance_risk import make_dataset, train


def test_model_learns_signal():
    pipe, metrics = train(make_dataset(n=3000, seed=1))
    # The synthetic signal is learnable; a fit model clears a weak baseline.
    assert metrics["roc_auc"] > 0.75
    assert metrics["n_test"] == 900


def test_predicts_probabilities():
    pipe, _ = train(make_dataset(n=1500, seed=2))
    sample = make_dataset(n=5, seed=9).drop(columns=["high_risk"])
    proba = pipe.predict_proba(sample)[:, 1]
    assert len(proba) == 5
    assert all(0.0 <= p <= 1.0 for p in proba)
