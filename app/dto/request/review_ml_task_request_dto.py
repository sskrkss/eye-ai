from pydantic import BaseModel

from models.enums import Diagnosis


class ReviewMlTaskRequestDto(BaseModel):
    doctor_conclusion: Diagnosis
