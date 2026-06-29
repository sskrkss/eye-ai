from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, status

from auth.authenticator import auth_super_admin
from database.database import get_session
from dto.request.super_admin.update_license_request_dto import UpdateLicenseRequestDto
from dto.response.admin_license_response_dto import AdminLicenseResponseDto
from dto.response.license_response_dto import LicenseResponseDto
from models import User
from services.license_service import LicenseService

admin_license_route = APIRouter()


@admin_license_route.get(
    "/",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=List[AdminLicenseResponseDto],
    status_code=status.HTTP_200_OK,
    summary="Get all licenses with company and scan usage"
)
async def get_all_licenses(
    admin: User = Depends(auth_super_admin),
    session=Depends(get_session)
) -> List[AdminLicenseResponseDto]:
    return LicenseService(session).get_all_licenses_info()


@admin_license_route.put(
    "/{license_id}",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=LicenseResponseDto,
    status_code=status.HTTP_200_OK,
    summary="Update license"
)
async def update_license(
    license_id: UUID,
    request_dto: UpdateLicenseRequestDto,
    admin: User = Depends(auth_super_admin),
    session=Depends(get_session)
) -> LicenseResponseDto:
    license_service = LicenseService(session)
    license = license_service.update_license(
        license_id=license_id,
        new_status=request_dto.status,
        expired_at=request_dto.expired_at,
        max_scans=request_dto.max_scans,
        notes=request_dto.notes
    )

    return LicenseResponseDto.model_validate(license)
