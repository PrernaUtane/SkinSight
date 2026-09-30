"""
SkinSight HTTP API.

Routes
------
GET  /              frontend
GET  /api/status    health + loaded model
POST /api/classify  lesion image -> ranked class scores
POST /api/card      predicted label -> educational JSON card
POST /api/chat      follow-up questions about the top predictions
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.config import (
    HF_MODEL_LABEL,
    HOST,
    INDEX_HTML,
    OPENROUTER_API_KEY,
    OPENROUTER_PRIMARY_MODEL,
    PORT,
    PROJECT_AUTHOR,
    env,
)
from app.inference import classifier
from app.llm_client import complete_chat, request_disease_card
from app.schemas import (
    CardRequest,
    ChatRequest,
    ChatResponse,
    ClassifyResponse,
    PredictionItem,
    StatusResponse,
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    skip = env("SKINSIGHT_SKIP_MODEL_LOAD", "").lower() in {"1", "true", "yes"}
    if skip:
        print("[SkinSight] skipping model load (SKINSIGHT_SKIP_MODEL_LOAD)")
    else:
        classifier.load()
    yield


app = FastAPI(
    title="SkinSight",
    description="Educational skin-lesion classification prototype by Prerna.",
    contact={"name": PROJECT_AUTHOR},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def frontend():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail=f"Missing UI file: {INDEX_HTML}")
    return FileResponse(INDEX_HTML)


@app.get("/api/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        status="ready" if classifier.ready else "loading",
        model=HF_MODEL_LABEL,
        device=str(classifier.device),
        llm=OPENROUTER_PRIMARY_MODEL,
        llm_configured=bool(OPENROUTER_API_KEY),
    )


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(image: UploadFile = File(...)):
    content_type = (image.content_type or "").lower()
    if content_type and content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WEBP image.",
        )

    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="The file could not be read as an image.",
        ) from exc

    try:
        ranked = classifier.predict(pil_image)
    except Exception as exc:
        print(f"[SkinSight] classify failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ClassifyResponse(
        predictions=[PredictionItem(label=item.label, score=item.score) for item in ranked],
        model=classifier.display_name,
    )


@app.post("/api/card")
async def disease_card(body: CardRequest):
    try:
        return request_disease_card(body.label, body.confidence)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[SkinSight] card failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    messages = [{"role": "system", "content": body.system_prompt}]
    messages.extend({"role": turn.role, "content": turn.content} for turn in body.history)
    try:
        reply = complete_chat(messages, temperature=0.7)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[SkinSight] chat failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(reply=reply or "I could not generate a reply. Please try again.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
