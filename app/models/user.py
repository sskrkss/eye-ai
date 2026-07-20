from typing import List, Optional
from uuid import UUID

from sqlalchemy import Column, JSON
from sqlalchemy.ext.mutable import MutableList
from sqlmodel import Field

from models.base_entity import BaseEntity
from models.enums import UserRole


class User(BaseEntity, table=True):
    email: str = Field(
        unique=True,
        index=True,
        min_length=5,
        max_length=255,
        nullable=False
    )
    username: str = Field(
        unique=True,
        index=True,
        min_length=5,
        max_length=255,
        nullable=False
    )
    first_name: str = Field(
        max_length=100,
        nullable=False
    )
    last_name: str = Field(
        max_length=100,
        nullable=False
    )
    middle_name: Optional[str] = Field(
        default=None,
        max_length=100,
        nullable=True
    )
    password_hash: str = Field(
        max_length=255,
        min_length=5,
        nullable=False
    )
    company_id: UUID = Field(
        foreign_key="company.id",
        nullable=False
    )
    roles: List[UserRole] = Field(
        default_factory=lambda: [UserRole.USER],
        sa_column=Column(
            MutableList.as_mutable(JSON),
            nullable=False
        )
    )

    def add_role(self, role: UserRole) -> None:
        if not self.has_role(role):
            self.roles.append(role)

    def remove_role(self, role: UserRole) -> None:
        if self.has_role(role):
            self.roles.remove(role)

    def has_role(self, role: UserRole) -> bool:
        return role in self.roles
