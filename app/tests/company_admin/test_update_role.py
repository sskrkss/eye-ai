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


def _set_role(client: TestClient, user_id, role: str):
    return client.put(f"/api/company-admin/users/{user_id}/role", json={"role": role})


def test_promote_to_admin(client: TestClient):
    user_id = _create_user(client, "promote@test.com", "promote_me").json()["id"]

    response = _set_role(client, user_id, "company_admin")

    assert response.status_code == 200
    assert "company_admin" in response.json()["roles"]


def test_demote_admin(client: TestClient):
    user_id = _create_user(client, "demote@test.com", "demote_me").json()["id"]
    _set_role(client, user_id, "company_admin")

    response = _set_role(client, user_id, "user")

    assert response.status_code == 200
    assert "company_admin" not in response.json()["roles"]


def test_change_own_role_forbidden(client: TestClient, test_user: User):
    response = _set_role(client, test_user.id, "user")

    assert response.status_code == 409
    assert response.json() == {"detail": "Cannot change your own role"}


def test_assign_super_admin_rejected(client: TestClient):
    user_id = _create_user(client, "super@test.com", "super_try").json()["id"]

    response = _set_role(client, user_id, "super_admin")

    assert response.status_code == 422


def test_change_role_other_company(client: TestClient, session: Session):
    other_company = Company(name="Other Company")
    session.add(other_company)
    session.commit()
    session.refresh(other_company)

    other_user = User(
        email="otherrole@test.com",
        username="other_role_user",
        first_name="Имя",
        last_name="Фамилия",
        password_hash=HashPassword().create_hash("qwerty123$"),
        company_id=other_company.id,
        roles=[UserRole.USER],
    )
    session.add(other_user)
    session.commit()
    session.refresh(other_user)

    response = _set_role(client, other_user.id, "company_admin")

    assert response.status_code == 404
    assert response.json() == {"detail": "User not found"}
