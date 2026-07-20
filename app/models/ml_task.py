import json
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship

from models.base_entity import BaseEntity
from models.enums import Diagnosis, TaskStatus

if TYPE_CHECKING:
    from models.patient import Patient


class MlTask(BaseEntity, table=True):
    prediction_json: Optional[str] = Field(
        default=None,
        sa_column=Column("prediction", JSON, nullable=True)
    )
    worker_id: Optional[str] = Field(
        default=None,
        nullable=True
    )
    task_status: TaskStatus = Field(
        default=TaskStatus.PROCESSING,
        nullable=False
    )
    finished_at: Optional[datetime] = Field(
        default=None,
        nullable=True
    )
    user_id: UUID = Field(
        foreign_key="user.id",
        nullable=False
    )
    patient_id: Optional[UUID] = Field(
        default=None,
        foreign_key="patient.id",
        nullable=True
    )
    image_filename: Optional[str] = Field(
        default=None,
        nullable=True
    )
    doctor_conclusion: Optional[Diagnosis] = Field(
        default=None,
        nullable=True
    )
    patient: Optional["Patient"] = Relationship(back_populates="ml_tasks")

    @property
    def prediction(self) -> Optional[dict]:
        if self.prediction_json:
            return json.loads(self.prediction_json)
        return None

    @prediction.setter
    def prediction(self, value: Optional[dict]):
        self.prediction_json = json.dumps(value) if value is not None else None
