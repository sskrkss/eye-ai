from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class UserResponseDto(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    username: str
    first_name: str
    last_name: str
    middle_name: Optional[str]
    roles: List[str]
