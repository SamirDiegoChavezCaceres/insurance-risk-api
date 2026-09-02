import pytest

from insurance_risk import create_app, make_dataset, train

VALID = {
    "age": 64, "bmi": 31.2, "num_chronic": 2, "exercise_days": 1,
    "sex": "M", "region": "south", "smoker": "yes",
}


@pytest.fixture(scope="module")
def client():
    pipe, _ = train(make_dataset(n=2000, seed=3))
    return create_app(pipe).test_client()


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_predict_returns_probability(client):
    res = client.post("/predict", json=VALID)
    assert res.status_code == 200
    body = res.get_json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_label"] in (0, 1)


def test_missing_field_is_400(client):
    bad = {k: v for k, v in VALID.items() if k != "smoker"}
    res = client.post("/predict", json=bad)
    assert res.status_code == 400
    assert "smoker" in res.get_json()["fields"]


def test_non_object_body_is_400(client):
    res = client.post("/predict", json=[1, 2, 3])
    assert res.status_code == 400
