from uuid import uuid4

from fastapi.testclient import TestClient

from models.license import License
from models.patient import Patient
from tests.conftest import make_test_image


def _run_task(client: TestClient, patient_id):
    return client.post(
        "/api/ml-tasks/run",
        files={"image": ("retina.jpg", make_test_image(), "image/jpeg")},
        data={"patient_id": str(patient_id)},
    )


def test_get_ml_task_by_id(client: TestClient, patient: Patient, license: License):
    task_id = _run_task(client, patient.id).json()["id"]

    response = client.get(f"/api/ml-tasks/{task_id}")

    assert response.status_code == 200
    assert response.json()["id"] == task_id


def test_get_ml_task_not_found(client: TestClient):
    response = client.get(f"/api/ml-tasks/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Ml task not found"
