from typing import Sequence
from uuid import UUID

from sqlmodel import select

from models.license import License
from repositories.base_repository import BaseRepository


class LicenseRepository(BaseRepository[License]):
    def get_by_id(self, id: UUID) -> License | None:
        return self._session.exec(select(License).where(License.id == id)).first()

    def get_by_company(self, company_id: UUID) -> License | None:
        return self._session.exec(select(License).where(License.company_id == company_id)).first()

    def get_all(self) -> Sequence[License]:
        return self._session.exec(select(License)).all()
