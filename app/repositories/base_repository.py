from abc import ABC
from sqlmodel import Session

from models.base_entity import BaseEntity


class BaseRepository[T: BaseEntity](ABC):
    def __init__(self, session: Session):
        self._session = session

    def save(self, entity: T) -> T:
        self._session.add(entity)
        self._session.commit()
        self._session.refresh(entity)

        return entity

    def delete(self, entity: T) -> None:
        self._session.delete(entity)
        self._session.commit()
