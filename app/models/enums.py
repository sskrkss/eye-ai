from enum import Enum


class UserRole(str, Enum):
    USER = "user"
    COMPANY_ADMIN = "company_admin"
    SUPER_ADMIN = "super_admin"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


class Diagnosis(str, Enum):
    UNKNOWN = "unknown"
    NO_DR = "no_dr"
    MILD_DR = "mild_dr"
    MODERATE_DR = "moderate_dr"
    SEVERE_DR = "severe_dr"
    PROLIFERATIVE_DR = "proliferative_dr"


class LicenseStatus(str, Enum):
    DEMO = "demo"
    ACTIVE = "active"
    EXPIRED = "expired"


class TaskStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEWED = "reviewed"
