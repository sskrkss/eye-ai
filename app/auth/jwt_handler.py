import time
from uuid import UUID

from fastapi import HTTPException, status
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError

from database.database import get_settings

settings = get_settings()
AUTH_SECRET_KEY = settings.AUTH_SECRET_KEY


def create_access_token(user_id: UUID) -> str:
    payload = {
        "user_id": str(user_id),
        "expires": time.time() + 3600
    }
    token = jwt.encode(payload, AUTH_SECRET_KEY, algorithm="HS256")

    return token


def verify_access_token(token: str) -> dict:
    try:
        decoded_token = jwt.decode(
            token=token,
            key=AUTH_SECRET_KEY,
            algorithms=["HS256"]
        )
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token expired"
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token"
        )

    return decoded_token
