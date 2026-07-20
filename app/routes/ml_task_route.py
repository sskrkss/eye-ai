from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from auth.authenticator import auth_user
from auth.s2s_authenticator import auth_ml_worker
from database.database import get_session
from dto.request.review_ml_task_request_dto import ReviewMlTaskRequestDto
from dto.request.save_ml_task_prediction_request_dto import SaveMlTaskPredictionDto
from dto.response.ml_task_response_dto import MlTaskResponseDto
from models import User
from services.ml_task_service import MlTaskService

ml_task_route = APIRouter()


@ml_task_route.post(
    "/run",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=MlTaskResponseDto,
    summary="Run DR detection on a retinal image",
    status_code=status.HTTP_200_OK
)
async def run_ml_task(
    image: UploadFile = File(...),
    patient_id: UUID = Form(...),
    user: User = Depends(auth_user),
    session=Depends(get_session)
) -> MlTaskResponseDto:
    image_bytes = await image.read()

    ml_task_service = MlTaskService(session)
    ml_task_service.validate_image(image.content_type, image_bytes)
    ml_task = ml_task_service.run_ml_task(user, image_bytes, patient_id, image.filename)

    return MlTaskResponseDto.model_validate(ml_task)


@ml_task_route.post(
    "/save-prediction",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=MlTaskResponseDto,
    summary="Save ml task prediction (only for s2s integration)",
    status_code=status.HTTP_200_OK
)
async def save_ml_task_prediction(
    request_dto: SaveMlTaskPredictionDto,
    s2s=Depends(auth_ml_worker),
    session=Depends(get_session)
) -> MlTaskResponseDto:
    ml_task_service = MlTaskService(session)
    ml_task = ml_task_service.save_ml_task_prediction(
        task_id=request_dto.task_id,
        task_status=request_dto.task_status,
        prediction=request_dto.prediction,
        worker_id=request_dto.worker_id
    )

    return MlTaskResponseDto.model_validate(ml_task)


@ml_task_route.put(
    "/{task_id}/review",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=MlTaskResponseDto,
    summary="Add doctor's conclusion and mark task as reviewed",
    status_code=status.HTTP_200_OK
)
async def review_ml_task(
    task_id: UUID,
    request_dto: ReviewMlTaskRequestDto,
    user: User = Depends(auth_user),
    session=Depends(get_session)
) -> MlTaskResponseDto:
    ml_task_service = MlTaskService(session)
    ml_task = ml_task_service.review_ml_task(task_id, user, request_dto.doctor_conclusion)

    return MlTaskResponseDto.model_validate(ml_task)


@ml_task_route.get(
    "/{task_id}",
    openapi_extra={"security": [{"BearerAuth": []}]},
    response_model=MlTaskResponseDto,
    summary="Get ml task by id",
    status_code=status.HTTP_200_OK
)
async def get_ml_task(
    task_id: UUID,
    user: User = Depends(auth_user),
    session=Depends(get_session)
) -> MlTaskResponseDto:
    ml_task_service = MlTaskService(session)
    ml_task = ml_task_service.get_ml_tasks_by_id_and_user(
        task_id=task_id,
        user=user
    )

    return MlTaskResponseDto.model_validate(ml_task)
