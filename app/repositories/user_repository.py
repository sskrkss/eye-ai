from uuid import UUID

from sqlmodel import select, Sequence

from models.user import User
from repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[User]):
    def get_by_id(self, id: UUID) -> User | None:
        return self._session.exec(select(User).where(User.id == id)).first()

    def get_by_id_and_company(self, id: UUID, company_id: UUID) -> User | None:
        return self._session.exec(
            select(User).where(User.id == id, User.company_id == company_id)
        ).first()

    def get_by_email(self, email: str) -> User | None:
        statement = select(User).where(User.email == email)

        return self._session.exec(statement).first()

    def get_by_username(self, username: str) -> User | None:
        statement = select(User).where(User.username == username)

        return self._session.exec(statement).first()

    def get_by_company(self, company_id: UUID) -> Sequence[User]:
        statement = select(User).where(User.company_id == company_id).order_by(User.username)

        return self._session.exec(statement).all()
