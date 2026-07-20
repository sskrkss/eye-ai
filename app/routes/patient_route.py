from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, status

from auth.authenticator import auth_user
from database.database import get_session
from dto.request.create_patient_request_dto import CreatePatientRequestDto
from dto.request.update_patient_request_dto import UpdatePatientRequestDto
from dto.response.ml_task_response_dto import MlTaskResponseDto
from dto.response.patient_response_dto import PatientResponseDto
from models import User
from repositories.ml_task_repository import MlTaskRepository
from services.patient_service import PatientService

patient_route = APIRouter()


@patient_route.post(
    "/",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=PatientResponseDto,
    status_code=status.HTTP_200_OK,
    summary="Create a new patient in the current clinic"
)
async def create_patient(
    request_dto: CreatePatientRequestDto,
    user: User = Depends(auth_user),
    session=Depends(get_session)
) -> PatientResponseDto:
    patient_service = PatientService(session)
    patient = patient_service.create_patient(
        user,
        first_name=request_dto.first_name,
        last_name=request_dto.last_name,
        age=request_dto.age,
        gender=request_dto.gender,
        middle_name=request_dto.middle_name
    )

    return PatientResponseDto.model_validate(patient)


@patient_route.put(
    "/{patient_id}",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=PatientResponseDto,
    status_code=status.HTTP_200_OK,
    summary="Update patient data"
)
async def update_patient(
    patient_id: UUID,
    request_dto: UpdatePatientRequestDto,
    user: User = Depends(auth_user),
    session=Depends(get_session)
) -> PatientResponseDto:
    patient_service = PatientService(session)
    patient = patient_service.update_patient(
        patient_id=patient_id,
        user=user,
        first_name=request_dto.first_name,
        middle_name=request_dto.middle_name,
        last_name=request_dto.last_name,
        age=request_dto.age,
        gender=request_dto.gender,
        current_diagnosis=request_dto.current_diagnosis
    )

    return PatientResponseDto.model_validate(patient)


@patient_route.get(
    "/",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=List[PatientResponseDto],
    status_code=status.HTTP_200_OK,
    summary="Get all patients of the current clinic"
)
async def get_patients(
    user: User = Depends(auth_user),
    session=Depends(get_session)
) -> List[PatientResponseDto]:
    patient_service = PatientService(session)

    return [
        PatientResponseDto.model_validate(p)
        for p in patient_service.get_patients(user)
    ]


@patient_route.get(
    "/{patient_id}",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=PatientResponseDto,
    status_code=status.HTTP_200_OK,
    summary="Get patient by id"
)
async def get_patient(
    patient_id: UUID,
    user: User = Depends(auth_user),
    session=Depends(get_session)
) -> PatientResponseDto:
    patient_service = PatientService(session)
    patient = patient_service.get_patient(patient_id, user)

    return PatientResponseDto.model_validate(patient)


@patient_route.get(
    "/{patient_id}/tasks",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=List[MlTaskResponseDto],
    status_code=status.HTTP_200_OK,
    summary="Get all DR screening tasks for a patient"
)
async def get_patient_tasks(
    patient_id: UUID,
    user: User = Depends(auth_user),
    session=Depends(get_session)
) -> List[MlTaskResponseDto]:
    patient_service = PatientService(session)
    patient = patient_service.get_patient(
        patient_id=patient_id,
        user=user
    )

    ml_task_repository = MlTaskRepository(session)
    tasks = ml_task_repository.get_by_patient(patient.id)

    return [
        MlTaskResponseDto.model_validate(t)
        for t in tasks
    ]
