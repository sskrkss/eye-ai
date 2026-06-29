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


def test_get_patient_tasks_empty(client: TestClient, patient: Patient):
    response = client.get(f"/api/patients/{patient.id}/tasks")

    assert response.status_code == 200
    assert response.json() == []


def test_get_patient_tasks(client: TestClient, patient: Patient, license: License):
    _run_task(client, patient.id)
    _run_task(client, patient.id)

    response = client.get(f"/api/patients/{patient.id}/tasks")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert all(t["task_status"] == "processing" for t in response.json())


def test_get_patient_tasks_not_found(client: TestClient):
    response = client.get(f"/api/patients/{uuid4()}/tasks")

    assert response.status_code == 404
    assert response.json()["detail"] == "Patient not found"
