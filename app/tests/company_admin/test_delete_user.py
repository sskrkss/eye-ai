from uuid import UUID

from fastapi.testclient import TestClient
from sqlmodel import select, Session

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


def test_delete_user(client: TestClient, session: Session, test_user: User):
    created = _create_user(client, "todelete@test.com", "to_delete")
    user_id = created.json()["id"]

    response = client.delete(f"/api/company-admin/users/{user_id}")
    assert response.status_code == 204

    found = session.exec(select(User).where(User.id == UUID(user_id))).first()
    assert found is None


def test_delete_user_self(client: TestClient, test_user: User):
    response = client.delete(f"/api/company-admin/users/{test_user.id}")
    assert response.status_code == 409
    assert response.json() == {"detail": "Cannot delete yourself"}


def test_delete_user_from_other_company(client: TestClient, session: Session):
    other_company = Company(name="Other Company")
    session.add(other_company)
    session.commit()
    session.refresh(other_company)

    other_user = User(
        email="other@test.com",
        username="other_user",
        first_name="Имя",
        last_name="Фамилия",
        password_hash=HashPassword().create_hash("qwerty123$"),
        company_id=other_company.id,
        roles=[UserRole.USER]
    )
    session.add(other_user)
    session.commit()
    session.refresh(other_user)

    response = client.delete(f"/api/company-admin/users/{other_user.id}")
    assert response.status_code == 404
    assert response.json() == {"detail": "User not found"}


def test_delete_user_not_found(client: TestClient):
    import uuid
    response = client.delete(f"/api/company-admin/users/{uuid.uuid4()}")
    assert response.status_code == 404
