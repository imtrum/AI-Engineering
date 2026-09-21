from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_roof():
    response = client.get("/")

    assert response.status_code == 200


def test_predict():
    data = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "prediction" in response.json()

def test_error_predict1():
    data = {
        "Pclass":3,
        "Sex": "male",
    }

    response = client.post("/predict", json= data)

    assert response.status_code == 422


def test_error_predict2():
    data = data = {
            "Pclass": 3,
            "Sex": "male",
            "Age": 22,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Embarked": "S"
        }

    response = client.post("/predict", json = data)
    assert isinstance(response.json()["prediction"], int)

