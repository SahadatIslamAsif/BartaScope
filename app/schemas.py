"""Request and response models for the Bartascope API."""

from typing import Dict, List

from pydantic import BaseModel, Field

from app.textproc import LABELS

MAX_CHARS = 2000


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS,
                      description="One SMS message.")


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64,
                             description="Up to 64 SMS messages.")


class Prediction(BaseModel):
    label: str = Field(..., description=f"One of {LABELS}")
    confidence: float = Field(..., description="Probability of the winning "
                                               "label.")
    probabilities: Dict[str, float] = Field(
        ..., description="Probability for all four classes; sums to 1.")
    language: str = Field(..., description="Heuristic script/word-list guess, "
                                           "not a language identifier. The "
                                           "model does not use it.")
    language_uncertain: bool = Field(
        ..., description="True when the language guess was a coin-flip.")
    chars: int = Field(..., description="Length of the submitted message.")
    model_version: str


class ClassifyResponse(Prediction):
    inference_ms: float = Field(..., description="Tokenisation plus ONNX "
                                                 "Runtime execution.")


class BatchClassifyResponse(BaseModel):
    results: List[Prediction]
    inference_ms: float = Field(..., description="For the whole batch.")
    count: int


class HealthResponse(BaseModel):
    status: str
    model_version: str


class ReadyResponse(BaseModel):
    status: str
    model_version: str
    model_path: str
    threads: int
    max_length: int
    load_seconds: float | None = None
