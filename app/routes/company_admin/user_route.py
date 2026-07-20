from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, status

from auth.authenticator import auth_company_admin
from database.database import get_session
from dto.request.create_user_request_dto import CreateUserRequestDto
from dto.request.update_user_role_request_dto import UpdateUserRoleRequestDto
from dto.response.user_response_dto import UserResponseDto
from models import User
from services.user_service import UserService

company_admin_user_route = APIRouter()


@company_admin_user_route.get(
    "/",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=List[UserResponseDto],
    status_code=status.HTTP_200_OK,
    summary="List users of the current company"
)
async def get_users(
    company_admin: User = Depends(auth_company_admin),
    session=Depends(get_session)
) -> List[UserResponseDto]:
    user_service = UserService(session)

    return [
        UserResponseDto.model_validate(user)
        for user in user_service.get_company_users(company_admin.company_id)
    ]


@company_admin_user_route.post(
    "/",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=UserResponseDto,
    status_code=status.HTTP_200_OK,
    summary="Create a new user in the current company"
)
async def create_user(
    request_dto: CreateUserRequestDto,
    company_admin: User = Depends(auth_company_admin),
    session=Depends(get_session)
) -> UserResponseDto:
    user_service = UserService(session)
    user = user_service.create_user(
        email=request_dto.email,
        username=request_dto.username,
        plain_password=request_dto.password,
        company_id=company_admin.company_id,
        first_name=request_dto.first_name,
        last_name=request_dto.last_name,
        middle_name=request_dto.middle_name
    )

    return UserResponseDto.model_validate(user)


@company_admin_user_route.delete(
    "/{user_id}",
    openapi_extra={"security": [{"BearerAuth": []}]},
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a user from the current company"
)
async def delete_user(
    user_id: UUID,
    company_admin: User = Depends(auth_company_admin),
    session=Depends(get_session)
) -> None:
    user_service = UserService(session)
    user_service.delete_user(company_admin, user_id)


@company_admin_user_route.put(
    "/{user_id}/role",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=UserResponseDto,
    status_code=status.HTTP_200_OK,
    summary="Change a user's role within the current company"
)
async def update_user_role(
    user_id: UUID,
    request_dto: UpdateUserRoleRequestDto,
    company_admin: User = Depends(auth_company_admin),
    session=Depends(get_session)
) -> UserResponseDto:
    user_service = UserService(session)
    user = user_service.set_user_role(company_admin, user_id, request_dto.role)

    return UserResponseDto.model_validate(user)
