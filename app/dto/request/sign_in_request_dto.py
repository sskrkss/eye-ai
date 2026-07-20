from pydantic import BaseModel


class SignInRequestDto(BaseModel):
    email_or_username: str
    plain_password: str
