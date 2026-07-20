from typing import Optional

from pydantic import BaseModel, field_validator

from models.enums import Gender


class CreatePatientRequestDto(BaseModel):
    first_name: str
    middle_name: Optional[str] = None
    last_name: str
    age: int
    gender: Gender

    @field_validator("first_name", "last_name")
    def validate_name(cls, value: str) -> str:
        v = value.strip()
        if len(v) < 1:
            raise ValueError("Must be at least 1 character")
        if len(v) > 100:
            raise ValueError("Must not exceed 100 characters")
        return v

    @field_validator("middle_name")
    def validate_middle_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = value.strip()
        return v if v else None

    @field_validator("age")
    def validate_age(cls, value: int) -> int:
        if value < 0 or value > 150:
            raise ValueError("Age must be between 0 and 150")
        return value
