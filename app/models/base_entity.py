from datetime import datetime
from typing import Optional
from uuid import uuid4, UUID
from sqlmodel import SQLModel, Field


class BaseEntity(SQLModel):
    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        nullable=False
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"onupdate": datetime.now}
    )
