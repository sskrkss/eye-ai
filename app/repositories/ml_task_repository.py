from typing import Sequence
from uuid import UUID

from sqlmodel import select, desc, func

from models import MlTask
from models.patient import Patient
from repositories.base_repository import BaseRepository


class MlTaskRepository(BaseRepository[MlTask]):
    def get_by_id(self, id: UUID) -> MlTask | None:
        return self._session.exec(select(MlTask).where(MlTask.id == id)).first()

    def count_by_company(self, company_id: UUID) -> int:
        return self._session.exec(
            select(func.count(MlTask.id))
            .join(Patient, MlTask.patient_id == Patient.id)
            .where(Patient.company_id == company_id)
        ).one()

    def get_by_patient(self, patient_id: UUID) -> Sequence[MlTask]:
        return self._session.exec(
            select(MlTask)
            .where(MlTask.patient_id == patient_id)
            .order_by(desc(MlTask.created_at))
        ).all()
