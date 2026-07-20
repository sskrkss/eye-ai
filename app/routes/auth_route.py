from fastapi import APIRouter, status, Depends, Response

from auth.cookie_util import set_auth_cookie, delete_auth_cookie
from auth.jwt_handler import create_access_token
from database.database import get_session
from dto.request.sign_in_request_dto import SignInRequestDto
from dto.request.sign_up_request_dto import SignUpRequestDto
from dto.response.message_response_dto import MessageResponseDto
from services.auth_service import AuthService

auth_route = APIRouter()


@auth_route.post(
    "/sign-up",
    response_model=MessageResponseDto,
    status_code=status.HTTP_200_OK,
    summary="User registration"
)
async def sign_up(
    request_dto: SignUpRequestDto,
    response: Response,
    session=Depends(get_session)
) -> MessageResponseDto:
    auth_service = AuthService(session)
    user = auth_service.sign_up(
        email=str(request_dto.email),
        username=request_dto.username,
        plain_password=request_dto.plain_password,
        company_name=request_dto.company_name,
        first_name=request_dto.first_name,
        last_name=request_dto.last_name,
        middle_name=request_dto.middle_name
    )

    access_token = create_access_token(user.id)

    set_auth_cookie(
        response=response,
        access_token=access_token
    )

    return MessageResponseDto(message="Registration successful")


@auth_route.post(
    "/sign-in",
    response_model=MessageResponseDto,
    status_code=status.HTTP_200_OK,
    summary="User login"
)
async def sign_in(
    request_dto: SignInRequestDto,
    response: Response,
    session=Depends(get_session)
) -> MessageResponseDto:
    auth_service = AuthService(session)
    user = auth_service.sign_in(
        email_or_username=request_dto.email_or_username,
        plain_password=request_dto.plain_password
    )

    access_token = create_access_token(user.id)

    set_auth_cookie(
        response=response,
        access_token=access_token
    )

    return MessageResponseDto(message="Login successful")


@auth_route.post(
    "/sign-out",
    response_model=MessageResponseDto,
    status_code=status.HTTP_200_OK,
    summary="User logout"
)
async def sign_out(response: Response) -> MessageResponseDto:
    delete_auth_cookie(response=response)

    return MessageResponseDto(message="Logout successful")
