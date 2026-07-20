from typing import Sequence
from uuid import UUID

from sqlmodel import select

from models.patient import Patient
from repositories.base_repository import BaseRepository


class PatientRepository(BaseRepository[Patient]):
    def get_by_id(self, id: UUID) -> Patient | None:
        return self._session.exec(select(Patient).where(Patient.id == id)).first()

    def get_by_company(self, company_id: UUID) -> Sequence[Patient]:
        return self._session.exec(
            select(Patient)
            .where(Patient.company_id == company_id)
            .order_by(Patient.last_name, Patient.first_name)
        ).all()

    def get_by_id_and_company(self, id: UUID, company_id: UUID) -> Patient | None:
        return self._session.exec(
            select(Patient).where(Patient.id == id, Patient.company_id == company_id)
        ).first()
