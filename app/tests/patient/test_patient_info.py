from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from models.company import Company
from models.enums import Diagnosis, Gender
from models.patient import Patient


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


def test_get_patients(client: TestClient):
    _create_patient(client, "Иван", "Иванов")
    _create_patient(client, "Пётр", "Петров")

    response = client.get("/api/patients/")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_patient_by_id(client: TestClient):
    created = _create_patient(client)
    patient_id = created.json()["id"]

    response = client.get(f"/api/patients/{patient_id}")

    assert response.status_code == 200
    assert response.json()["id"] == patient_id


def test_get_patient_from_other_company(client: TestClient, session: Session):
    other_company = Company(name="Other Clinic")
    session.add(other_company)
    session.commit()
    session.refresh(other_company)

    other_patient = Patient(
        first_name="Чужой",
        last_name="Пациент",
        age=50,
        gender=Gender.MALE,
        current_diagnosis=Diagnosis.UNKNOWN,
        company_id=other_company.id
    )
    session.add(other_patient)
    session.commit()
    session.refresh(other_patient)

    response = client.get(f"/api/patients/{other_patient.id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Patient not found"


def test_get_patient_not_found(client: TestClient):
    response = client.get(f"/api/patients/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Patient not found"
