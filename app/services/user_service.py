from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlmodel import Session, Sequence

from auth.hash_password_util import HashPassword
from models.enums import UserRole
from models.user import User
from repositories.user_repository import UserRepository


class UserService:
    def __init__(self, session: Session):
        self.user_repository = UserRepository(session)

    def create_user(
        self,
        email: str,
        username: str,
        plain_password: str,
        company_id: UUID,
        first_name: str,
        last_name: str,
        middle_name: Optional[str] = None
    ) -> User:
        if self.user_repository.get_by_email(email) or self.user_repository.get_by_username(username):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email or username already exists"
            )

        user = User(
            email=email,
            username=username,
            first_name=first_name,
            last_name=last_name,
            middle_name=middle_name,
            password_hash=HashPassword().create_hash(plain_password),
            company_id=company_id,
            roles=[UserRole.USER]
        )

        return self.user_repository.save(user)

    def get_company_users(self, company_id: UUID) -> Sequence[User]:
        return self.user_repository.get_by_company(company_id)

    def set_user_role(self, company_admin: User, target_user_id: UUID, role: UserRole) -> User:
        if company_admin.id == target_user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot change your own role"
            )

        user = self.user_repository.get_by_id_and_company(target_user_id, company_admin.company_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        if role == UserRole.COMPANY_ADMIN:
            user.add_role(UserRole.COMPANY_ADMIN)
        else:
            user.remove_role(UserRole.COMPANY_ADMIN)

        return self.user_repository.save(user)

    def delete_user(self, company_admin: User, target_user_id: UUID) -> None:
        if company_admin.id == target_user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete yourself"
            )

        user = self.user_repository.get_by_id_and_company(target_user_id, company_admin.company_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        self.user_repository.delete(user)

    def get_user_by_id(self, id: UUID) -> User:
        user = self.user_repository.get_by_id(id)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return user

    def add_super_admin_role(self, user: User) -> User:
        user.add_role(UserRole.SUPER_ADMIN)
        return self.user_repository.save(user)

    def add_company_admin_role(self, user: User) -> User:
        user.add_role(UserRole.COMPANY_ADMIN)
        return self.user_repository.save(user)
