from uuid import UUID

from fastapi import Request, Depends, HTTPException, status
from fastapi.security.utils import get_authorization_scheme_param
from sqlalchemy.orm.session import Session

from auth.cookie_util import get_auth_cookie
from auth.jwt_handler import verify_access_token
from database.database import get_session
from models import User
from models.enums import UserRole
from services.user_service import UserService


class Authenticator:
    def __init__(self, required_role: UserRole):
        self.required_role = required_role

    async def __call__(
            self,
            request: Request,
            session: Session = Depends(get_session)
    ) -> User:
        auth_cookie = get_auth_cookie(request)

        if not auth_cookie:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )

        scheme, token = get_authorization_scheme_param(auth_cookie)
        decoded_token = verify_access_token(token)

        user_service = UserService(session)

        try:
            user = user_service.get_user_by_id(UUID(decoded_token["user_id"]))
        except HTTPException:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )

        if not user.has_role(self.required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {self.required_role.value}"
            )

        return user


auth_user = Authenticator(required_role=UserRole.USER)
auth_company_admin = Authenticator(required_role=UserRole.COMPANY_ADMIN)
auth_super_admin = Authenticator(required_role=UserRole.SUPER_ADMIN)
