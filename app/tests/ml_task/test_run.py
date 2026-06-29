from fastapi.testclient import TestClient
from jose import jwt
from sqlmodel import Session

from auth.s2s_authenticator import S2S_SECRET_KEY
from models.enums import LicenseStatus
from models.license import License
from models.patient import Patient
from tests.conftest import make_test_image


def _run_task(client: TestClient, patient_id, fmt="JPEG", content_type="image/jpeg"):
    return client.post(
        "/api/ml-tasks/run",
        files={"image": ("retina.jpg", make_test_image(fmt), content_type)},
        data={"patient_id": str(patient_id)},
    )


def _s2s_token() -> str:
    return jwt.encode(
        claims={"iss": "ml_worker", "sub": "app"},
        key=S2S_SECRET_KEY,
        algorithm="HS256",
    )


def test_ml_task_run(client: TestClient, patient: Patient, license: License):
    response1 = _run_task(client, patient.id)

    assert response1.status_code == 200
    assert response1.json()["task_status"] == "processing"
    assert response1.json()["image_filename"] == "retina.jpg"

    response2 = client.get(f"/api/ml-tasks/{response1.json()['id']}")
    assert response2.status_code == 200
    assert response2.json() == response1.json()

    response3 = client.post(
        "/api/ml-tasks/save-prediction",
        json={
            "task_id": response1.json()["id"],
            "task_status": "completed",
            "prediction": {"score": 1.42, "diagnosis": "mild_dr"},
            "worker_id": "test-worker",
        },
        headers={"Authorization": f"Bearer {_s2s_token()}"},
    )
    assert response3.status_code == 200

    response4 = client.get(f"/api/ml-tasks/{response1.json()['id']}")
    assert response4.status_code == 200
    assert response4.json()["task_status"] == "completed"
    assert response4.json()["prediction"] == {"score": 1.42, "diagnosis": "mild_dr"}
    assert response4.json()["worker_id"] == "test-worker"


def test_ml_task_run_png(client: TestClient, patient: Patient, license: License):
    response = _run_task(client, patient.id, fmt="PNG", content_type="image/png")
    assert response.status_code == 200


def test_ml_task_run_invalid_content_type(client: TestClient, patient: Patient):
    response = client.post(
        "/api/ml-tasks/run",
        files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        data={"patient_id": str(patient.id)},
    )
    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]


def test_ml_task_run_no_license(client: TestClient, session: Session, patient: Patient, license: License):
    session.delete(license)
    session.commit()

    response = _run_task(client, patient.id)
    assert response.status_code == 403
    assert response.json()["detail"] == "No active license"


def test_ml_task_run_does_not_update_patient_diagnosis(
    client: TestClient, session: Session, patient: Patient, license: License
):
    response1 = _run_task(client, patient.id)
    assert response1.status_code == 200

    client.post(
        "/api/ml-tasks/save-prediction",
        json={
            "task_id": response1.json()["id"],
            "task_status": "completed",
            "prediction": {"score": 1.42, "diagnosis": "mild_dr"},
            "worker_id": "test-worker",
        },
        headers={"Authorization": f"Bearer {_s2s_token()}"},
    )

    session.refresh(patient)
    assert patient.current_diagnosis == "unknown"


def test_ml_task_run_demo_limit(client: TestClient, session: Session, patient: Patient, license: License):
    license.status = LicenseStatus.DEMO
    license.max_scans = 2
    session.add(license)
    session.commit()

    _run_task(client, patient.id)
    _run_task(client, patient.id)

    response = _run_task(client, patient.id)
    assert response.status_code == 403
    assert response.json()["detail"] == f"Demo scan limit of {license.max_scans} reached. Please upgrade your license"


def test_ml_task_run_expired_license(client: TestClient, session: Session, patient: Patient, license: License):
    license.status = LicenseStatus.EXPIRED
    session.add(license)
    session.commit()

    response = _run_task(client, patient.id)
    assert response.status_code == 403
    assert response.json()["detail"] == "License has expired"


def test_ml_task_run_scan_limit(client: TestClient, session: Session, patient: Patient, license: License):
    license.status = LicenseStatus.ACTIVE
    license.max_scans = 1
    session.add(license)
    session.commit()

    _run_task(client, patient.id)

    response = _run_task(client, patient.id)
    assert response.status_code == 403
    assert response.json()["detail"] == f"Scan limit of {license.max_scans} reached. Please contact your administrator"


def test_ml_task_run_unlimited(client: TestClient, session: Session, patient: Patient, license: License):
    license.status = LicenseStatus.ACTIVE
    license.max_scans = None
    session.add(license)
    session.commit()

    _run_task(client, patient.id)
    _run_task(client, patient.id)

    response = _run_task(client, patient.id)
    assert response.status_code == 200


def test_ml_task_run_image_too_large(client: TestClient, patient: Patient):
    large_image = b"0" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/api/ml-tasks/run",
        files={"image": ("retina.jpg", large_image, "image/jpeg")},
        data={"patient_id": str(patient.id)},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Image size exceeds 10 MB limit"
