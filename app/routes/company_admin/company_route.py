from fastapi import APIRouter, Depends, status

from auth.authenticator import auth_company_admin
from database.database import get_session
from dto.response.company_response_dto import CompanyResponseDto
from models import User
from services.company_service import CompanyService

company_admin_company_route = APIRouter()


@company_admin_company_route.get(
    "/",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=CompanyResponseDto,
    status_code=status.HTTP_200_OK,
    summary="Get current company info"
)
async def get_company(
    admin: User = Depends(auth_company_admin),
    session=Depends(get_session)
) -> CompanyResponseDto:
    return CompanyService(session).get_company_info(admin.company_id)
