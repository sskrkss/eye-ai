from fastapi import Request, HTTPException, status
from fastapi.security.utils import get_authorization_scheme_param
from jose import jwt
from jose.exceptions import JWTError

from database.config import get_settings

settings = get_settings()
S2S_SECRET_KEY = settings.S2S_SECRET_KEY


class S2SAuthenticator:
    def __init__(self, issuer: str):
        self.issuer = issuer
        self.subject = "app"

    async def __call__(self, request: Request) -> None:
        s2s_token = request.headers.get("Authorization")

        if not s2s_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )

        scheme, token = get_authorization_scheme_param(s2s_token)

        try:
            jwt.decode(
                token=token,
                key=S2S_SECRET_KEY,
                algorithms=["HS256"],
                issuer=self.issuer,
                subject=self.subject
            )
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token"
            )


auth_ml_worker = S2SAuthenticator(issuer="ml_worker")
