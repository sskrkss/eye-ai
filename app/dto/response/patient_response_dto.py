from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from models.enums import Diagnosis, Gender


class PatientResponseDto(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    first_name: str
    middle_name: Optional[str]
    last_name: str
    age: int
    gender: Gender
    current_diagnosis: Diagnosis
    company_id: UUID
    created_at: datetime
