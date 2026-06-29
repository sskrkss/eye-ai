from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from models.enums import Diagnosis, TaskStatus


class MlTaskResponseDto(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    task_status: TaskStatus
    created_at: datetime
    finished_at: Optional[datetime] = None
    prediction: Optional[dict] = None
    worker_id: Optional[str] = None
    image_filename: Optional[str] = None
    doctor_conclusion: Optional[Diagnosis] = None
