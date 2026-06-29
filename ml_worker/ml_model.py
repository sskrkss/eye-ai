import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

MODEL_PATH = os.getenv("MODEL_PATH")
if MODEL_PATH is None:
    raise RuntimeError("Env variable MODEL_PATH is not set")


def _get_threshold(name: str) -> float:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Env variable {name} is not set")
    return float(value)


def _score_to_diagnosis(score: float) -> str:
    threshold_mild_dr = _get_threshold("THRESHOLD_MILD_DR")
    threshold_moderate_dr = _get_threshold("THRESHOLD_MODERATE_DR")
    threshold_severe_dr = _get_threshold("THRESHOLD_SEVERE_DR")
    threshold_proliferative_dr = _get_threshold("THRESHOLD_PROLIFERATIVE_DR")

    if score >= threshold_proliferative_dr:
        return "proliferative_dr"
    if score >= threshold_severe_dr:
        return "severe_dr"
    if score >= threshold_moderate_dr:
        return "moderate_dr"
    if score >= threshold_mild_dr:
        return "mild_dr"
    return "no_dr"


class MlModel:
    _instance = None
    _model = None
    _transform = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._model, self._transform = self._load_model()

    def _load_model(self):
        model = efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(1280, 1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        transform = EfficientNet_V2_S_Weights.DEFAULT.transforms()

        return model, transform

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        nparr = np.frombuffer(image_bytes, np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")

        # Crop to retinal area, pad to square, resize to 384px
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            img = img[y:y + h, x:x + w]

        h, w = img.shape[:2]
        side = max(h, w)
        padded = np.zeros((side, side, 3), dtype=np.uint8)
        padded[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = img
        img = cv2.resize(padded, (384, 384), interpolation=cv2.INTER_LANCZOS4)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return self._transform(pil_img).unsqueeze(0)

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> dict:
        tensor = self._preprocess(image_bytes)
        score = self._model(tensor).item()

        return {
            "score": round(score, 4),
            "diagnosis": _score_to_diagnosis(score)
        }
