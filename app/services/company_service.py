from uuid import UUID

from fastapi import HTTPException, status
from sqlmodel import Session

from dto.response.company_response_dto import CompanyResponseDto
from repositories.company_repository import CompanyRepository
from repositories.ml_task_repository import MlTaskRepository
from services.license_service import LicenseService


class CompanyService:
    def __init__(self, session: Session):
        self.company_repository = CompanyRepository(session)
        self.ml_task_repository = MlTaskRepository(session)
        self.license_service = LicenseService(session)

    def get_company_info(self, company_id: UUID) -> CompanyResponseDto:
        company = self.company_repository.get_by_id(company_id)
        if company is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Company not found"
            )

        license = self.license_service.get_license_by_company(company_id)
        license = self.license_service.refresh_status(license)

        used_scans = self.ml_task_repository.count_by_company(company_id)
        remaining_scans = None if license.max_scans is None else max(license.max_scans - used_scans, 0)

        return CompanyResponseDto(
            name=company.name,
            license_status=license.status,
            max_scans=license.max_scans,
            used_scans=used_scans,
            remaining_scans=remaining_scans,
            activated_at=license.activated_at,
            expired_at=license.expired_at,
        )
