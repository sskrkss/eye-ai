from typing import Optional, Sequence
from uuid import UUID

from fastapi import HTTPException, status
from sqlmodel import Session

from models.enums import Diagnosis, Gender
from models.patient import Patient
from models.user import User
from repositories.patient_repository import PatientRepository


class PatientService:
    def __init__(self, session: Session):
        self.patient_repository = PatientRepository(session)

    def create_patient(
        self,
        user: User,
        first_name: str,
        last_name: str,
        age: int,
        gender: Gender,
        middle_name: Optional[str] = None
    ) -> Patient:
        patient = Patient(
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            age=age,
            gender=gender,
            company_id=user.company_id
        )

        return self.patient_repository.save(patient)

    def update_patient(
        self,
        patient_id: UUID,
        user: User,
        first_name: Optional[str] = None,
        middle_name: Optional[str] = None,
        last_name: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[Gender] = None,
        current_diagnosis: Optional[Diagnosis] = None
    ) -> Patient:
        patient = self.get_patient(patient_id, user)

        if first_name is not None:
            patient.first_name = first_name
        if middle_name is not None:
            patient.middle_name = middle_name
        if last_name is not None:
            patient.last_name = last_name
        if age is not None:
            patient.age = age
        if gender is not None:
            patient.gender = gender
        if current_diagnosis is not None:
            patient.current_diagnosis = current_diagnosis

        return self.patient_repository.save(patient)

    def get_patients(self, user: User) -> Sequence[Patient]:
        return self.patient_repository.get_by_company(user.company_id)

    def get_patient(self, patient_id: UUID, user: User) -> Patient:
        patient = self.patient_repository.get_by_id_and_company(patient_id, user.company_id)

        if patient is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )

        return patient
