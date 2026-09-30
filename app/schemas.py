"""Request and response shapes for the HTTP API."""

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    predictions: list[PredictionItem]
    model: str


class StatusResponse(BaseModel):
    status: str
    model: str
    device: str
    llm: str
    llm_configured: bool


class CardRequest(BaseModel):
    label: str
    confidence: float


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    system_prompt: str
    history: list[ChatTurn]


class ChatResponse(BaseModel):
    reply: str
