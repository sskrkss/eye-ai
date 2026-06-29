from typing import Optional

from pydantic import BaseModel, EmailStr, field_validator


class SignUpRequestDto(BaseModel):
    email: EmailStr
    username: str
    plain_password: str
    company_name: str
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

    @field_validator("username")
    def validate_username(cls, value: str) -> str:
        if len(value) < 5:
            raise ValueError("Username must be at least 5 characters")
        if len(value) > 50:
            raise ValueError("Username must not exceed 50 characters")
        return value

    @field_validator("plain_password")
    def validate_password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters")
        return value

    @field_validator("company_name")
    def validate_company_name(cls, value: str) -> str:
        v = value.strip()
        if len(v) < 2:
            raise ValueError("Company name must be at least 2 characters")
        if len(v) > 255:
            raise ValueError("Company name must not exceed 255 characters")
        return v
