from pydantic import BaseModel, field_validator

from models.enums import UserRole


class UpdateUserRoleRequestDto(BaseModel):
    role: UserRole

    @field_validator("role")
    def validate_role(cls, value: UserRole) -> UserRole:
        if value == UserRole.SUPER_ADMIN:
            raise ValueError("Cannot assign super admin role")
        return value
