from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlmodel import Session

from models.company import Company
from models.enums import LicenseStatus
from models.license import License
from models.patient import Patient
from tests.conftest import make_test_image


def _run_task(client: TestClient, patient_id):
    return client.post(
        "/api/ml-tasks/run",
        files={"image": ("retina.jpg", make_test_image(), "image/jpeg")},
        data={"patient_id": str(patient_id)},
    )


def test_get_company_info(client: TestClient, company: Company, license: License):
    response = client.get("/api/company-admin/companies/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == company.name
    assert body["license_status"] == "active"
    assert body["max_scans"] == 100
    assert body["used_scans"] == 0
    assert body["remaining_scans"] == 100


def test_get_company_info_counts_used_scans(client: TestClient, patient: Patient, license: License):
    _run_task(client, patient.id)
    _run_task(client, patient.id)

    response = client.get("/api/company-admin/companies/")

    assert response.status_code == 200
    body = response.json()
    assert body["used_scans"] == 2
    assert body["remaining_scans"] == body["max_scans"] - 2


def test_get_company_info_no_license(client: TestClient, company: Company):
    response = client.get("/api/company-admin/companies/")

    assert response.status_code == 404
    assert response.json()["detail"] == "License not found"


def test_get_company_info_lazy_expiry(client: TestClient, session: Session, company: Company):
    license = License(
        company_id=company.id,
        status=LicenseStatus.ACTIVE,
        expired_at=datetime.now() - timedelta(days=1),
        max_scans=100,
    )
    session.add(license)
    session.commit()

    response = client.get("/api/company-admin/companies/")

    assert response.status_code == 200
    assert response.json()["license_status"] == "expired"
