from uuid import UUID

from sqlmodel import select

from models.company import Company
from repositories.base_repository import BaseRepository


class CompanyRepository(BaseRepository[Company]):
    def get_by_id(self, id: UUID) -> Company | None:
        return self._session.exec(select(Company).where(Company.id == id)).first()

    def get_by_name(self, name: str) -> Company | None:
        return self._session.exec(select(Company).where(Company.name == name)).first()
