from fastapi import APIRouter, Depends, status

from auth.authenticator import auth_user
from dto.response.user_response_dto import UserResponseDto
from models import User

user_route = APIRouter()


@user_route.get(
    "/current",
    openapi_extra={
        "security": [{"BearerAuth": []}],
    },
    response_model=UserResponseDto,
    status_code=status.HTTP_200_OK,
    summary="Get current user info",
    response_description="Current user info"
)
async def get_current_user(user: User = Depends(auth_user)) -> UserResponseDto:
    return UserResponseDto.model_validate(user)
