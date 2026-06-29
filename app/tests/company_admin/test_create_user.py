from fastapi.testclient import TestClient
from sqlmodel import select, Session

from models import User


def _create_user(client: TestClient, email: str, username: str, password: str = "qwerty123$"):
    return client.post(
        "/api/company-admin/users/",
        json={"email": email, "username": username, "password": password,
              "first_name": "Имя", "last_name": "Фамилия"}
    )


def test_create_user(client: TestClient, session: Session):
    response = _create_user(client, "newdoctor@test.com", "new_doctor")

    assert response.status_code == 200

    found_user = session.exec(select(User).where(User.email == "newdoctor@test.com")).first()
    assert found_user is not None
    assert found_user.username == "new_doctor"
    assert found_user.last_name == "Фамилия"
    assert found_user.first_name == "Имя"
    assert response.json()["last_name"] == "Фамилия"


def test_create_user_missing_name(client: TestClient):
    response = client.post(
        "/api/company-admin/users/",
        json={"email": "noname@test.com", "username": "no_name_doc", "password": "qwerty123$"}
    )

    assert response.status_code == 422


def test_create_user_duplicate_email(client: TestClient):
    _create_user(client, "dup@test.com", "doctor_one")
    response = _create_user(client, "dup@test.com", "doctor_two")

    assert response.status_code == 409
    assert response.json() == {"detail": "User with this email or username already exists"}


def test_create_user_duplicate_username(client: TestClient):
    _create_user(client, "doc1@test.com", "same_doctor")
    response = _create_user(client, "doc2@test.com", "same_doctor")

    assert response.status_code == 409
    assert response.json() == {"detail": "User with this email or username already exists"}


def test_create_user_short_password(client: TestClient):
    response = _create_user(client, "doc3@test.com", "doctor_three", password="123")
    assert response.status_code == 422


def test_create_user_invalid_email(client: TestClient):
    response = _create_user(client, "notanemail", "doctor_four")
    assert response.status_code == 422


def test_create_user_short_username(client: TestClient):
    response = _create_user(client, "doc5@test.com", "ab")
    assert response.status_code == 422
