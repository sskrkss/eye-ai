from fastapi.testclient import TestClient

from models import User


def test_get_current_user(client: TestClient, test_user: User):
    response = client.get("/api/users/current")

    assert response.status_code == 200
    assert response.json()["id"] == str(test_user.id)
    assert response.json()["email"] == test_user.email
    assert response.json()["username"] == test_user.username
