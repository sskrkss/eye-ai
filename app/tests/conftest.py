import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from api import app
from auth.authenticator import auth_company_admin, auth_super_admin, auth_user
from auth.hash_password_util import HashPassword
from database.database import get_session
from models import User
from models.company import Company
from models.enums import Diagnosis, Gender, LicenseStatus, UserRole
from models.license import License
from models.patient import Patient


def make_test_image(fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(200, 100, 50)).save(buf, format=fmt)

    return buf.getvalue()


@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine("sqlite:///testing.db", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture(name="company")
def company_fixture(session: Session) -> Company:
    company = Company(name="Test Company")
    session.add(company)
    session.commit()
    session.refresh(company)

    return company


@pytest.fixture(name="license")
def license_fixture(session: Session, company: Company) -> License:
    license = License(company_id=company.id, status=LicenseStatus.ACTIVE, max_scans=100)
    session.add(license)
    session.commit()
    session.refresh(license)

    return license


@pytest.fixture(name="test_user")
def test_user_fixture(session: Session, company: Company) -> User:
    user = User(
        email="auth_user@test.ru",
        username="test_auth_user",
        first_name="Тест",
        last_name="Тестов",
        password_hash=HashPassword().create_hash("qwerty123$"),
        company_id=company.id,
        roles=[UserRole.USER, UserRole.COMPANY_ADMIN]
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    return user


@pytest.fixture(name="patient")
def patient_fixture(session: Session, company: Company) -> Patient:
    patient = Patient(
        first_name="Иван",
        last_name="Иванов",
        age=45,
        gender=Gender.MALE,
        current_diagnosis=Diagnosis.UNKNOWN,
        company_id=company.id
    )
    session.add(patient)
    session.commit()
    session.refresh(patient)

    return patient


@pytest.fixture(name="client")
def client_fixture(session: Session, test_user: User):
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[auth_user] = lambda: test_user
    app.dependency_overrides[auth_company_admin] = lambda: test_user

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(name="super_admin_user")
def super_admin_user_fixture(session: Session, company: Company) -> User:
    user = User(
        email="super_admin@test.ru",
        username="super_admin",
        first_name="Супер",
        last_name="Админов",
        password_hash=HashPassword().create_hash("qwerty123$"),
        company_id=company.id,
        roles=[UserRole.SUPER_ADMIN]
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    return user


@pytest.fixture(name="admin_client")
def admin_client_fixture(session: Session, super_admin_user: User):
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[auth_super_admin] = lambda: super_admin_user

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
