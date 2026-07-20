from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from models.enums import LicenseStatus


class LicenseResponseDto(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    company_id: UUID
    status: LicenseStatus
    activated_at: Optional[datetime]
    expired_at: Optional[datetime]
    max_scans: Optional[int]
    notes: Optional[str]
    created_at: datetime
