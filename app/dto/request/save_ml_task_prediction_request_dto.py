from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from models.enums import TaskStatus


class SaveMlTaskPredictionDto(BaseModel):
    task_id: UUID
    task_status: TaskStatus
    prediction: Optional[dict]
    worker_id: str
