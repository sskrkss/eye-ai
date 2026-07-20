from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator

from models.enums import LicenseStatus


class UpdateLicenseRequestDto(BaseModel):
    status: Optional[LicenseStatus] = None
    expired_at: Optional[datetime] = None
    max_scans: Optional[int] = None
    notes: Optional[str] = None

    @field_validator("max_scans")
    def validate_positive(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("Must be a positive integer")
        return value
