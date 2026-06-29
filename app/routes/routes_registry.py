from fastapi import APIRouter

from routes.company_admin.company_route import company_admin_company_route
from routes.company_admin.user_route import company_admin_user_route
from routes.super_admin.license_route import admin_license_route
from routes.auth_route import auth_route
from routes.health_check_route import health_check_route
from routes.ml_task_route import ml_task_route
from routes.patient_route import patient_route
from routes.user_route import user_route
from routes.web_route import web_route


def get_app_router() -> APIRouter:
    router = APIRouter()

    router.include_router(health_check_route, tags=['Health check'])
    router.include_router(auth_route, prefix='/api', tags=['Auth'])
    router.include_router(user_route, prefix='/api/users', tags=['User'])
    router.include_router(company_admin_company_route, prefix='/api/company-admin/companies', tags=['Company admin'])
    router.include_router(company_admin_user_route, prefix='/api/company-admin/users', tags=['Company admin'])
    router.include_router(admin_license_route, prefix='/api/admin/licenses', tags=['Super admin'])
    router.include_router(ml_task_route, prefix='/api/ml-tasks', tags=['Ml task'])
    router.include_router(patient_route, prefix='/api/patients', tags=['Patient'])
    router.include_router(web_route, tags=['Web'])

    return router
