from uuid import uuid4

from fastapi.testclient import TestClient
from jose import jwt

from auth.s2s_authenticator import S2S_SECRET_KEY
from models.license import License
from models.patient import Patient
from tests.conftest import make_test_image


def _run_task(client: TestClient, patient_id):
    return client.post(
        "/api/ml-tasks/run",
        files={"image": ("retina.jpg", make_test_image(), "image/jpeg")},
        data={"patient_id": str(patient_id)},
    )


def _complete_task(client: TestClient, task_id: str):
    s2s_token = jwt.encode(
        claims={"iss": "ml_worker", "sub": "app"},
        key=S2S_SECRET_KEY,
        algorithm="HS256",
    )
    return client.post(
        "/api/ml-tasks/save-prediction",
        json={
            "task_id": task_id,
            "task_status": "completed",
            "prediction": {"score": 0.87, "diagnosis": "mild_dr"},
            "worker_id": "test-worker",
        },
        headers={"Authorization": f"Bearer {s2s_token}"},
    )


def test_save_prediction_failed(client: TestClient, patient: Patient, license: License):
    task_id = _run_task(client, patient.id).json()["id"]

    s2s_token = jwt.encode(
        claims={"iss": "ml_worker", "sub": "app"},
        key=S2S_SECRET_KEY,
        algorithm="HS256",
    )
    response = client.post(
        "/api/ml-tasks/save-prediction",
        json={
            "task_id": task_id,
            "task_status": "failed",
            "prediction": None,
            "worker_id": "test-worker",
        },
        headers={"Authorization": f"Bearer {s2s_token}"},
    )

    assert response.status_code == 200
    assert response.json()["task_status"] == "failed"


def test_save_prediction_unknown_task(client: TestClient):
    response = _complete_task(client, str(uuid4()))

    assert response.status_code == 404
    assert response.json()["detail"] == "Ml task not found"


def test_review_ml_task(client: TestClient, patient: Patient, license: License):
    task_id = _run_task(client, patient.id).json()["id"]
    _complete_task(client, task_id)

    response = client.put(
        f"/api/ml-tasks/{task_id}/review",
        json={"doctor_conclusion": "no_dr"},
    )

    assert response.status_code == 200
    assert response.json()["task_status"] == "reviewed"
    assert response.json()["doctor_conclusion"] == "no_dr"


def test_review_ml_task_re_review(client: TestClient, patient: Patient, license: License):
    task_id = _run_task(client, patient.id).json()["id"]
    _complete_task(client, task_id)

    first = client.put(f"/api/ml-tasks/{task_id}/review", json={"doctor_conclusion": "no_dr"})
    assert first.status_code == 200
    assert first.json()["doctor_conclusion"] == "no_dr"

    second = client.put(f"/api/ml-tasks/{task_id}/review", json={"doctor_conclusion": "mild_dr"})

    assert second.status_code == 200
    assert second.json()["task_status"] == "reviewed"
    assert second.json()["doctor_conclusion"] == "mild_dr"


def test_review_ml_task_not_completed(client: TestClient, patient: Patient, license: License):
    task_id = _run_task(client, patient.id).json()["id"]

    response = client.put(
        f"/api/ml-tasks/{task_id}/review",
        json={"doctor_conclusion": "no_dr"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Only completed tasks can be reviewed"


def test_review_ml_task_invalid_conclusion(client: TestClient, patient: Patient, license: License):
    task_id = _run_task(client, patient.id).json()["id"]
    _complete_task(client, task_id)

    response = client.put(
        f"/api/ml-tasks/{task_id}/review",
        json={"doctor_conclusion": "not_a_valid_diagnosis"},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "doctor_conclusion"]
