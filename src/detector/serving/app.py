"""POST an image, get back an AI-generation score in [0, 1] (0.5 = decision threshold)."""

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps

from detector.serving.detector import Detector

MODELS = os.getenv("MODELS", "models")
MAX_BYTES = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector(MODELS)  # missing model files fail startup loudly
    yield


app = FastAPI(title="AI image detector", lifespan=lifespan)


def _score(data: bytes) -> float:
    try:
        with Image.open(io.BytesIO(data)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        return app.state.detector.score(image)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, f"Unreadable or too small image: {exc}") from exc


@app.post("/predict")
async def predict(file: UploadFile) -> dict[str, float]:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 20 MiB")
    return {"ai_score": await run_in_threadpool(_score, data)}
