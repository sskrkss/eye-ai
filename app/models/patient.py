from typing import List, Optional, TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from models.base_entity import BaseEntity
from models.enums import Diagnosis, Gender

if TYPE_CHECKING:
    from models.ml_task import MlTask


class Patient(BaseEntity, table=True):
    first_name: str = Field(
        max_length=100,
        nullable=False
    )
    middle_name: Optional[str] = Field(
        default=None,
        max_length=100,
        nullable=True
    )
    last_name: str = Field(
        max_length=100,
        nullable=False
    )
    age: int = Field(
        nullable=False
    )
    gender: Gender = Field(
        nullable=False
    )
    current_diagnosis: Diagnosis = Field(
        default=Diagnosis.UNKNOWN,
        nullable=False
    )
    company_id: UUID = Field(
        foreign_key="company.id",
        nullable=False
    )
    ml_tasks: List["MlTask"] = Relationship(
        back_populates="patient",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "lazy": "selectin"
        }
    )
