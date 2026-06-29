from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from models.enums import LicenseStatus


class AdminLicenseResponseDto(BaseModel):
    id: UUID
    company_id: UUID
    company_name: str
    status: LicenseStatus
    max_scans: Optional[int]
    used_scans: int
    activated_at: Optional[datetime]
    expired_at: Optional[datetime]
    notes: Optional[str]
