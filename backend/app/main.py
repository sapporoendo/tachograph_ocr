from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .detect_disc import detect_disc


class Segment(BaseModel):
    start: str
    end: str
    type: Literal["driving", "stop", "unknown"]
    confidence: Literal["high", "mid", "low"]


class AnalyzeResponse(BaseModel):
    totalDrivingMinutes: int
    totalStopMinutes: int
    needsReviewMinutes: int
    segments: list[Segment]


app = FastAPI(title="tachograph_ocr backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    chartType: Optional[str] = Form(None),
    midnightOffsetDeg: Optional[float] = Form(None),
) -> AnalyzeResponse:
    _ = chartType
    _ = midnightOffsetDeg
    image_bytes = await file.read()
    detect_disc(image_bytes)

    return AnalyzeResponse(
        totalDrivingMinutes=390,
        totalStopMinutes=80,
        needsReviewMinutes=12,
        segments=[
            Segment(start="08:00", end="12:30", type="driving", confidence="high"),
            Segment(start="12:30", end="13:10", type="stop", confidence="mid"),
            Segment(start="13:10", end="17:00", type="driving", confidence="high"),
            Segment(start="17:00", end="17:12", type="unknown", confidence="low"),
        ],
    )
