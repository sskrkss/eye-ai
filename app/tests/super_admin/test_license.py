from uuid import uuid4

from fastapi.testclient import TestClient

from models.license import License


def test_get_all_licenses(admin_client: TestClient, license: License):
    response = admin_client.get("/api/admin/licenses/")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["id"] == str(license.id)


def test_update_license_activate(admin_client: TestClient, license: License):
    response = admin_client.put(
        f"/api/admin/licenses/{license.id}",
        json={"status": "active"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "active"
    assert response.json()["activated_at"] is not None


def test_update_license_max_scans(admin_client: TestClient, license: License):
    response = admin_client.put(
        f"/api/admin/licenses/{license.id}",
        json={"max_scans": 500},
    )

    assert response.status_code == 200
    assert response.json()["max_scans"] == 500


def test_update_license_notes(admin_client: TestClient, license: License):
    response = admin_client.put(
        f"/api/admin/licenses/{license.id}",
        json={"notes": "Клиника Здоровье, договор №42"},
    )

    assert response.status_code == 200
    assert response.json()["notes"] == "Клиника Здоровье, договор №42"


def test_update_license_not_found(admin_client: TestClient):
    response = admin_client.put(
        f"/api/admin/licenses/{uuid4()}",
        json={"status": "active"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "License not found"


def test_update_license_invalid_max_scans(admin_client: TestClient, license: License):
    response = admin_client.put(
        f"/api/admin/licenses/{license.id}",
        json={"max_scans": 0},
    )

    assert response.status_code == 422
