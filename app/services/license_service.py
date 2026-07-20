from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlmodel import Session

from dto.response.admin_license_response_dto import AdminLicenseResponseDto
from models.enums import LicenseStatus
from models.license import License
from repositories.company_repository import CompanyRepository
from repositories.license_repository import LicenseRepository
from repositories.ml_task_repository import MlTaskRepository


class LicenseService:
    def __init__(self, session: Session):
        self.license_repository = LicenseRepository(session)
        self.ml_task_repository = MlTaskRepository(session)
        self.company_repository = CompanyRepository(session)

    def update_license(
        self,
        license_id: UUID,
        new_status: Optional[LicenseStatus] = None,
        expired_at: Optional[datetime] = None,
        max_scans: Optional[int] = None,
        notes: Optional[str] = None
    ) -> License:
        license = self.license_repository.get_by_id(license_id)
        if license is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="License not found"
            )

        if new_status is not None:
            if new_status == LicenseStatus.ACTIVE and license.activated_at is None:
                license.activated_at = datetime.now()
            license.status = new_status
        if expired_at is not None:
            license.expired_at = expired_at
        if max_scans is not None:
            license.max_scans = max_scans
        if notes is not None:
            license.notes = notes

        return self.license_repository.save(license)

    def get_license_by_company(self, company_id: UUID) -> License:
        license = self.license_repository.get_by_company(company_id)

        if license is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="License not found"
            )

        return license

    def get_all_licenses_info(self) -> List[AdminLicenseResponseDto]:
        result = []
        for license in self.license_repository.get_all():
            license = self.refresh_status(license)
            company = self.company_repository.get_by_id(license.company_id)

            result.append(AdminLicenseResponseDto(
                id=license.id,
                company_id=license.company_id,
                company_name=company.name if company else "—",
                status=license.status,
                max_scans=license.max_scans,
                used_scans=self.ml_task_repository.count_by_company(license.company_id),
                activated_at=license.activated_at,
                expired_at=license.expired_at,
                notes=license.notes,
            ))
        return result

    def refresh_status(self, license: License) -> License:
        if (license.status == LicenseStatus.ACTIVE
                and license.expired_at is not None
                and license.expired_at < datetime.now()):
            license.status = LicenseStatus.EXPIRED
            self.license_repository.save(license)

        return license

    def check_license(self, company_id: UUID) -> None:
        license = self.license_repository.get_by_company(company_id)

        if license is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No active license"
            )

        license = self.refresh_status(license)

        if license.status == LicenseStatus.EXPIRED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="License has expired"
            )

        if license.max_scans is None:
            return

        scan_count = self.ml_task_repository.count_by_company(company_id)
        if scan_count >= license.max_scans:
            if license.status == LicenseStatus.DEMO:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Demo scan limit of {license.max_scans} reached. Please upgrade your license"
                )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scan limit of {license.max_scans} reached. Please contact your administrator"
            )
