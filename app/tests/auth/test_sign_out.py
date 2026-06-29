from fastapi.testclient import TestClient

from auth.cookie_util import AUTH_COOKIE_NAME


def test_sign_out(client: TestClient):
    client.post(
        "/api/sign-up",
        json={
            "email": "user@test.com",
            "username": "test_user",
            "plain_password": "qwerty123$",
            "company_name": "Clinic One",
            "first_name": "Имя",
            "last_name": "Фамилия",
        }
    )

    client.post(
        "/api/sign-in",
        json={"email_or_username": "user@test.com", "plain_password": "qwerty123$"}
    )

    response = client.post("/api/sign-out")

    assert response.status_code == 200
    assert response.json() == {"message": "Logout successful"}
    assert response.cookies.get(AUTH_COOKIE_NAME) is None
