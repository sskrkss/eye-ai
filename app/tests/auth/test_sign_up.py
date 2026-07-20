from fastapi.security.utils import get_authorization_scheme_param
from fastapi.testclient import TestClient
from sqlmodel import select, Session

from auth.cookie_util import AUTH_COOKIE_NAME
from auth.hash_password_util import HashPassword
from auth.jwt_handler import verify_access_token
from models import User
from models.company import Company
from models.enums import LicenseStatus
from models.license import License


def _sign_up(client: TestClient, email: str, username: str, company_name: str, password: str = "qwerty123$"):
    return client.post(
        "/api/sign-up",
        json={
            "email": email,
            "username": username,
            "plain_password": password,
            "company_name": company_name,
            "first_name": "Имя",
            "last_name": "Фамилия",
        }
    )


def test_sign_up(client: TestClient, session: Session):
    response = _sign_up(client, "user@test.com", "test_user", "Clinic One")

    assert response.status_code == 200
    assert response.json() == {"message": "Registration successful"}

    found_user = session.exec(select(User).where(User.email == "user@test.com")).first()

    assert found_user.email == "user@test.com"
    assert found_user.username == "test_user"
    assert found_user.last_name == "Фамилия"
    assert found_user.first_name == "Имя"
    assert HashPassword().verify_hash("qwerty123$", found_user.password_hash)

    cookie_value = response.cookies.get(AUTH_COOKIE_NAME)
    scheme, token = get_authorization_scheme_param(cookie_value)
    payload = verify_access_token(token.strip('"'))

    assert payload["user_id"] == str(found_user.id)

    demo_license = session.exec(select(License).where(License.company_id == found_user.company_id)).first()
    assert demo_license is not None
    assert demo_license.status == LicenseStatus.DEMO
    assert demo_license.company_id == found_user.company_id


def test_sign_up_not_unique_username(client: TestClient):
    _sign_up(client, "user1@test.com", "test_user1", "Clinic One")
    response = _sign_up(client, "user2@test.com", "test_user1", "Clinic Two")

    assert response.status_code == 409
    assert response.json() == {"detail": "User with this email or username already exists"}


def test_sign_up_not_unique_email(client: TestClient):
    _sign_up(client, "user3@test.com", "test_user3", "Clinic Three")
    response = _sign_up(client, "user3@test.com", "test_user4", "Clinic Four")

    assert response.status_code == 409
    assert response.json() == {"detail": "User with this email or username already exists"}


def test_sign_up_not_unique_company(client: TestClient):
    _sign_up(client, "user5@test.com", "test_user5", "Same Clinic")
    response = _sign_up(client, "user6@test.com", "test_user6", "Same Clinic")

    assert response.status_code == 409
    assert response.json() == {"detail": "Company with this name already exists"}


def test_sign_up_duplicate_user_does_not_create_company(client: TestClient, session: Session):
    _sign_up(client, "dup@test.com", "dup_user", "First Clinic")

    response = _sign_up(client, "dup@test.com", "another_user", "Brand New Clinic")

    assert response.status_code == 409
    orphan = session.exec(select(Company).where(Company.name == "Brand New Clinic")).first()
    assert orphan is None


def test_sign_up_missing_name(client: TestClient):
    response = client.post(
        "/api/sign-up",
        json={
            "email": "noname@test.com",
            "username": "no_name_user",
            "plain_password": "qwerty123$",
            "company_name": "Nameless Clinic",
        }
    )

    assert response.status_code == 422


def test_sign_up_not_valid_email_format(client: TestClient):
    response = _sign_up(client, "usertest.com", "test_user7", "Clinic Seven")

    assert response.status_code == 422
    assert (response.json()["detail"][0]["msg"] ==
            "value is not a valid email address: An email address must have an @-sign.")


def test_sign_up_short_username(client: TestClient):
    response = _sign_up(client, "user8@test.com", "test", "Clinic Eight")

    assert response.status_code == 422
    assert (response.json()["detail"][0]["msg"] ==
            "Value error, Username must be at least 5 characters")


def test_sign_up_long_username(client: TestClient):
    response = _sign_up(client, "user9@test.com", "t" * 51, "Clinic Nine")

    assert response.status_code == 422
    assert (response.json()["detail"][0]["msg"] ==
            "Value error, Username must not exceed 50 characters")


def test_sign_up_short_password(client: TestClient):
    response = _sign_up(client, "user10@test.com", "test_user10", "Clinic Ten", password="qwerty")

    assert response.status_code == 422
    assert (response.json()["detail"][0]["msg"] ==
            "Value error, Password must be at least 8 characters")


def test_sign_up_short_company_name(client: TestClient):
    response = _sign_up(client, "user11@test.com", "test_user11", "A")

    assert response.status_code == 422
    assert (response.json()["detail"][0]["msg"] ==
            "Value error, Company name must be at least 2 characters")


def test_sign_up_long_company_name(client: TestClient):
    response = _sign_up(client, "user12@test.com", "test_user12", "C" * 256)

    assert response.status_code == 422
    assert (response.json()["detail"][0]["msg"] ==
            "Value error, Company name must not exceed 255 characters")
