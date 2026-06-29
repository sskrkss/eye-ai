from fastapi.testclient import TestClient

from models.patient import Patient


def test_update_patient_name(client: TestClient, patient: Patient):
    response = client.put(
        f"/api/patients/{patient.id}",
        json={"first_name": "Пётр", "last_name": "Сидоров"},
    )

    assert response.status_code == 200
    assert response.json()["first_name"] == "Пётр"
    assert response.json()["last_name"] == "Сидоров"
    assert response.json()["age"] == patient.age


def test_update_patient_diagnosis(client: TestClient, patient: Patient):
    response = client.put(
        f"/api/patients/{patient.id}",
        json={"current_diagnosis": "mild_dr"},
    )

    assert response.status_code == 200
    assert response.json()["current_diagnosis"] == "mild_dr"


def test_update_patient_invalid_diagnosis(client: TestClient, patient: Patient):
    response = client.put(
        f"/api/patients/{patient.id}",
        json={"current_diagnosis": "not_valid"},
    )

    assert response.status_code == 422


def test_update_patient_invalid_age(client: TestClient, patient: Patient):
    response = client.put(
        f"/api/patients/{patient.id}",
        json={"age": 200},
    )

    assert response.status_code == 422


def test_update_patient_not_found(client: TestClient):
    import uuid
    response = client.put(
        f"/api/patients/{uuid.uuid4()}",
        json={"first_name": "Иван"},
    )

    assert response.status_code == 404


def test_update_patient_from_other_company(client: TestClient, session, patient: Patient):
    from models.company import Company
    from models.enums import Diagnosis, Gender
    from models.patient import Patient as PatientModel

    other_company = Company(name="Other Clinic")
    session.add(other_company)
    session.commit()
    session.refresh(other_company)

    other_patient = PatientModel(
        first_name="Чужой",
        last_name="Пациент",
        age=40,
        gender=Gender.MALE,
        current_diagnosis=Diagnosis.UNKNOWN,
        company_id=other_company.id,
    )
    session.add(other_patient)
    session.commit()
    session.refresh(other_patient)

    response = client.put(
        f"/api/patients/{other_patient.id}",
        json={"first_name": "Хакер"},
    )

    assert response.status_code == 404
