from fastapi import Response, Request

from database.config import get_settings

settings = get_settings()
AUTH_COOKIE_NAME = settings.AUTH_COOKIE_NAME


def get_auth_cookie(request: Request) -> str | None:
    return request.cookies.get(AUTH_COOKIE_NAME)


def set_auth_cookie(response: Response, access_token: str) -> None:
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=f"Bearer {access_token}",
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=3600,
        path="/"
    )


def delete_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path="/"
    )
