from fastapi.testclient import TestClient


def _create_patient(client: TestClient, first_name="Иван", last_name="Иванов", age=45, gender="male"):
    return client.post(
        "/api/patients/",
        json={
            "first_name": first_name,
            "last_name": last_name,
            "age": age,
            "gender": gender,
        }
    )


def test_create_patient(client: TestClient):
    response = _create_patient(client)

    assert response.status_code == 200
    assert response.json()["first_name"] == "Иван"
    assert response.json()["last_name"] == "Иванов"
    assert response.json()["age"] == 45
    assert response.json()["gender"] == "male"
    assert response.json()["current_diagnosis"] == "unknown"


def test_create_patient_with_middle_name(client: TestClient):
    response = client.post(
        "/api/patients/",
        json={
            "first_name": "Анна",
            "middle_name": "Сергеевна",
            "last_name": "Петрова",
            "age": 38,
            "gender": "female",
        }
    )

    assert response.status_code == 200
    assert response.json()["middle_name"] == "Сергеевна"


def test_create_patient_invalid_age(client: TestClient):
    response = _create_patient(client, age=200)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "age"]
