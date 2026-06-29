from fastapi.testclient import TestClient
from sqlmodel import Session

from auth.hash_password_util import HashPassword
from models import User
from models.company import Company
from models.enums import UserRole


def _create_user(client: TestClient, email: str, username: str, password: str = "qwerty123$"):
    return client.post(
        "/api/company-admin/users/",
        json={"email": email, "username": username, "password": password,
              "first_name": "Имя", "last_name": "Фамилия"}
    )


def test_list_users(client: TestClient, test_user: User):
    _create_user(client, "doc1@test.com", "doctor_one")
    _create_user(client, "doc2@test.com", "doctor_two")

    response = client.get("/api/company-admin/users/")

    assert response.status_code == 200
    usernames = {u["username"] for u in response.json()}
    assert {"test_auth_user", "doctor_one", "doctor_two"} <= usernames


def test_list_users_scoped_to_company(client: TestClient, session: Session):
    other_company = Company(name="Other Company")
    session.add(other_company)
    session.commit()
    session.refresh(other_company)

    other_user = User(
        email="other@test.com",
        username="other_company_user",
        first_name="Имя",
        last_name="Фамилия",
        password_hash=HashPassword().create_hash("qwerty123$"),
        company_id=other_company.id,
        roles=[UserRole.USER],
    )
    session.add(other_user)
    session.commit()

    response = client.get("/api/company-admin/users/")

    assert response.status_code == 200
    usernames = {u["username"] for u in response.json()}
    assert "other_company_user" not in usernames
