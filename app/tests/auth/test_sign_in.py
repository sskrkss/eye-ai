from fastapi.security.utils import get_authorization_scheme_param
from fastapi.testclient import TestClient
from sqlmodel import select, Session

from auth.cookie_util import AUTH_COOKIE_NAME
from auth.jwt_handler import verify_access_token
from models import User


def _sign_up(client: TestClient, email: str, username: str, company_name: str):
    client.post(
        "/api/sign-up",
        json={
            "email": email,
            "username": username,
            "plain_password": "qwerty123$",
            "company_name": company_name,
            "first_name": "Имя",
            "last_name": "Фамилия",
        }
    )


def test_sign_in_by_email(client: TestClient, session: Session):
    _sign_up(client, "user@test.com", "test_user", "Clinic One")

    response = client.post(
        "/api/sign-in",
        json={"email_or_username": "user@test.com", "plain_password": "qwerty123$"}
    )

    assert response.status_code == 200
    assert response.json() == {"message": "Login successful"}

    found_user = session.exec(select(User).where(User.email == "user@test.com")).first()

    cookie_value = response.cookies.get(AUTH_COOKIE_NAME)
    scheme, token = get_authorization_scheme_param(cookie_value)
    payload = verify_access_token(token.strip('"'))

    assert payload["user_id"] == str(found_user.id)


def test_sign_in_by_username(client: TestClient, session: Session):
    _sign_up(client, "user1@test.com", "test_user1", "Clinic Two")

    response = client.post(
        "/api/sign-in",
        json={"email_or_username": "test_user1", "plain_password": "qwerty123$"}
    )

    assert response.status_code == 200
    assert response.json() == {"message": "Login successful"}

    found_user = session.exec(select(User).where(User.username == "test_user1")).first()

    cookie_value = response.cookies.get(AUTH_COOKIE_NAME)
    scheme, token = get_authorization_scheme_param(cookie_value)
    payload = verify_access_token(token.strip('"'))

    assert payload["user_id"] == str(found_user.id)


def test_sign_in_not_found(client: TestClient):
    response = client.post(
        "/api/sign-in",
        json={"email_or_username": "nobody", "plain_password": "qwerty123$"}
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid credentials"}
    assert response.cookies.get(AUTH_COOKIE_NAME) is None


def test_sign_in_wrong_password(client: TestClient):
    _sign_up(client, "user3@test.com", "test_user3", "Clinic Three")

    response = client.post(
        "/api/sign-in",
        json={"email_or_username": "test_user3", "plain_password": "wrongpassword"}
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid credentials"}
    assert response.cookies.get(AUTH_COOKIE_NAME) is None
