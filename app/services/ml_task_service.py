from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlmodel import Session

from models.enums import Diagnosis, TaskStatus
from models.ml_task import MlTask
from models.user import User
from repositories.ml_task_repository import MlTaskRepository
from repositories.patient_repository import PatientRepository
from services.license_service import LicenseService
from services.rmq.ml_task_publisher import send_task

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/tiff"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


class MlTaskService:
    def __init__(self, session: Session):
        self.ml_task_repository = MlTaskRepository(session)
        self.patient_repository = PatientRepository(session)
        self.license_service = LicenseService(session)

    def validate_image(self, content_type: str, image_bytes: bytes) -> None:
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported file type: {content_type}. Allowed: JPEG, PNG, TIFF"
            )

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Image size exceeds 10 MB limit"
            )

    def run_ml_task(
        self,
        user: User,
        image_bytes: bytes,
        patient_id: UUID,
        image_filename: Optional[str] = None
    ) -> MlTask:
        self.license_service.check_license(user.company_id)

        patient = self.patient_repository.get_by_id_and_company(patient_id, user.company_id)
        if patient is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )

        ml_task = MlTask(user_id=user.id, patient_id=patient.id, image_filename=image_filename)
        self.ml_task_repository.save(ml_task)

        send_task(ml_task.id, image_bytes)

        return ml_task

    def save_ml_task_prediction(
        self,
        task_id: UUID,
        task_status: TaskStatus,
        prediction: dict | None,
        worker_id: str
    ) -> MlTask:
        ml_task = self.ml_task_repository.get_by_id(task_id)

        if ml_task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ml task not found"
            )

        if task_status == TaskStatus.COMPLETED:
            ml_task.prediction = prediction
            ml_task.task_status = TaskStatus.COMPLETED

        elif task_status == TaskStatus.FAILED:
            ml_task.task_status = TaskStatus.FAILED

        ml_task.finished_at = datetime.now()
        ml_task.worker_id = worker_id

        self.ml_task_repository.save(ml_task)

        return ml_task

    def review_ml_task(self, task_id: UUID, user: User, doctor_conclusion: Diagnosis) -> MlTask:
        ml_task = self.get_ml_tasks_by_id_and_user(task_id, user)

        if ml_task.task_status not in (TaskStatus.COMPLETED, TaskStatus.REVIEWED):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only completed tasks can be reviewed"
            )

        ml_task.doctor_conclusion = doctor_conclusion
        ml_task.task_status = TaskStatus.REVIEWED

        return self.ml_task_repository.save(ml_task)

    def get_ml_tasks_by_id_and_user(self, task_id: UUID, user: User) -> MlTask:
        ml_task = self.ml_task_repository.get_by_id(task_id)

        if ml_task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ml task not found"
            )

        patient = self.patient_repository.get_by_id_and_company(ml_task.patient_id, user.company_id)
        if patient is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Ml task is not permitted"
            )

        return ml_task
