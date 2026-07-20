from pydantic import BaseModel


class MessageResponseDto(BaseModel):
    message: str
