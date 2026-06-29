from typing import Optional

from pydantic import BaseModel, field_validator


class CreateUserRequestDto(BaseModel):
    email: str
    username: str
    password: str
    first_name: str
    last_name: str
    middle_name: Optional[str] = None

    @field_validator("first_name", "last_name")
    def validate_name(cls, value: str) -> str:
        v = value.strip()
        if len(v) < 1:
            raise ValueError("Name is required")
        if len(v) > 100:
            raise ValueError("Name must not exceed 100 characters")
        return v

    @field_validator("middle_name")
    def validate_middle_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = value.strip()
        if len(v) == 0:
            return None
        if len(v) > 100:
            raise ValueError("Name must not exceed 100 characters")
        return v

    @field_validator("email")
    def validate_email(cls, value: str) -> str:
        v = value.strip().lower()
        if "@" not in v:
            raise ValueError("Invalid email")
        return v

    @field_validator("username")
    def validate_username(cls, value: str) -> str:
        v = value.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        return v

    @field_validator("password")
    def validate_password(cls, value: str) -> str:
        if len(value) < 6:
            raise ValueError("Password must be at least 6 characters")
        return value
