from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.orm.session import Session

from auth.hash_password_util import HashPassword
from models import User
from models.company import Company
from models.license import License
from repositories.company_repository import CompanyRepository
from repositories.license_repository import LicenseRepository
from repositories.user_repository import UserRepository
from services.user_service import UserService


class AuthService:
    def __init__(self, session: Session):
        self.user_repository = UserRepository(session)
        self.company_repository = CompanyRepository(session)
        self.license_repository = LicenseRepository(session)
        self.user_service = UserService(session)

    def sign_up(
        self,
        email: str,
        username: str,
        plain_password: str,
        company_name: str,
        first_name: str,
        last_name: str,
        middle_name: Optional[str] = None
    ) -> User:
        if self.company_repository.get_by_name(company_name):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Company with this name already exists"
            )

        if self.user_repository.get_by_email(email) or self.user_repository.get_by_username(username):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email or username already exists"
            )

        company = self.company_repository.save(Company(name=company_name))

        self.license_repository.save(License(company_id=company.id))

        user = self.user_service.create_user(
            email=email,
            username=username,
            plain_password=plain_password,
            company_id=company.id,
            first_name=first_name,
            last_name=last_name,
            middle_name=middle_name
        )
        user = self.user_service.add_company_admin_role(user)

        return user

    def sign_in(self, email_or_username: str, plain_password: str) -> User:
        user = self.user_repository.get_by_email(email_or_username)

        if user is None:
            user = self.user_repository.get_by_username(email_or_username)

        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        hash_password = HashPassword()
        if not hash_password.verify_hash(plain_password, user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        return user
