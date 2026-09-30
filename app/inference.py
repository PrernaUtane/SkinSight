"""
Single-image classification.

The checkpoint is a Hugging Face image classifier. We use that model's
own AutoImageProcessor so preprocessing matches how it was trained and
how the evaluation notebook scores it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.config import HF_MODEL_ID, HF_MODEL_LABEL


@dataclass
class Prediction:
    label: str
    score: float


class LesionClassifier:
    def __init__(self, model_id: str = HF_MODEL_ID, display_name: str = HF_MODEL_LABEL):
        self.model_id = model_id
        self.display_name = display_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor: Optional[AutoImageProcessor] = None
        self.model = None

    def load(self) -> None:
        print(f"[SkinSight] device = {self.device}")
        print(f"[SkinSight] loading {self.model_id}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        print("[SkinSight] classifier ready")

    @property
    def ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def predict(self, image: Image.Image) -> list[Prediction]:
        if not self.ready:
            raise RuntimeError("Classifier is not loaded yet.")

        rgb = image.convert("RGB")
        encoded = self.processor(images=rgb, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits

        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        id_to_label = self.model.config.id2label

        ranked = [
            Prediction(label=id_to_label.get(index, f"class_{index}"), score=float(score))
            for index, score in enumerate(probabilities)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked


classifier = LesionClassifier()
