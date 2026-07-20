from sqlmodel import Field

from models.base_entity import BaseEntity


class Company(BaseEntity, table=True):
    name: str = Field(
        max_length=255,
        nullable=False,
        unique=True,
        index=True
    )
