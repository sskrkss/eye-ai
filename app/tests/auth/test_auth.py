from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from api import app
from auth.cookie_util import AUTH_COOKIE_NAME
from auth.hash_password_util import HashPassword
from auth.jwt_handler import create_access_token
from database.database import get_session
from models import User
from models.company import Company
from models.enums import UserRole


@pytest.fixture(name="api_client")
def api_client_fixture(session: Session):
    """Client with the real authenticator active — only the DB session is overridden."""
    app.dependency_overrides[get_session] = lambda: session
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def _auth(client: TestClient, user: User) -> None:
    client.cookies.clear()
    client.cookies.set(AUTH_COOKIE_NAME, f"Bearer {create_access_token(user.id)}")


def test_no_cookie_returns_401(api_client: TestClient):
    api_client.cookies.clear()
    response = api_client.get("/api/users/current")

    assert response.status_code == 401


def test_invalid_token_rejected(api_client: TestClient):
    api_client.cookies.clear()
    api_client.cookies.set(AUTH_COOKIE_NAME, "Bearer not-a-jwt")
    response = api_client.get("/api/users/current")

    assert response.status_code == 400


def test_unknown_user_returns_401(api_client: TestClient):
    api_client.cookies.clear()
    api_client.cookies.set(AUTH_COOKIE_NAME, f"Bearer {create_access_token(uuid4())}")
    response = api_client.get("/api/users/current")

    assert response.status_code == 401


def test_valid_token_authenticates(api_client: TestClient, test_user: User):
    _auth(api_client, test_user)
    response = api_client.get("/api/users/current")

    assert response.status_code == 200
    assert response.json()["email"] == test_user.email


def test_wrong_role_forbidden(api_client: TestClient, session: Session, company: Company):
    user = User(
        email="plain@test.com",
        username="plain_user",
        first_name="Имя",
        last_name="Фамилия",
        password_hash=HashPassword().create_hash("qwerty123$"),
        company_id=company.id,
        roles=[UserRole.USER],
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    _auth(api_client, user)
    response = api_client.get("/api/company-admin/users/")

    assert response.status_code == 403


def test_correct_role_allowed(api_client: TestClient, test_user: User):
    _auth(api_client, test_user)
    response = api_client.get("/api/company-admin/users/")

    assert response.status_code == 200
