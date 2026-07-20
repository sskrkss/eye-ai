from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlmodel import Field

from models.base_entity import BaseEntity
from models.enums import LicenseStatus


class License(BaseEntity, table=True):
    company_id: UUID = Field(
        foreign_key="company.id",
        nullable=False,
        unique=True,
        index=True
    )
    status: LicenseStatus = Field(
        default=LicenseStatus.DEMO,
        nullable=False
    )
    activated_at: Optional[datetime] = Field(
        default=None,
        nullable=True
    )
    expired_at: Optional[datetime] = Field(
        default=None,
        nullable=True
    )
    max_scans: Optional[int] = Field(
        default=10,
        nullable=True
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        nullable=True
    )
