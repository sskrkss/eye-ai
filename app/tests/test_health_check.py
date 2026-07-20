from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    response = client.get("/health-check")
    assert response.status_code == 200
    assert response.json() == {"message": "Health check successful"}
