from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from models.enums import LicenseStatus


class CompanyResponseDto(BaseModel):
    name: str
    license_status: LicenseStatus
    max_scans: Optional[int]
    used_scans: int
    remaining_scans: Optional[int]
    activated_at: Optional[datetime]
    expired_at: Optional[datetime]
