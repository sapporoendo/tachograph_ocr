from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .analyze import analyze_image, failure_response

app = FastAPI(title="tachograph_ocr api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/analyze")
async def analyze(
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    chartType: Optional[str] = Form(default=None),
    midnightOffsetDeg: Optional[float] = Form(default=None),
) -> dict:
    up = file or image
    if up is None:
        return failure_response(
            error_code="NO_FILE",
            message="No file provided. Send multipart field 'file' (preferred) or 'image'.",
            hint="Flutter is expected to send field name 'file'.",
        )

    try:
        raw = await up.read()
        if not raw:
            return failure_response(
                error_code="EMPTY_FILE",
                message="Uploaded file is empty.",
                hint="Please re-upload the image.",
            )

        return analyze_image(
            image_bytes=raw,
            chart_type=chartType,
            midnight_offset_deg=midnightOffsetDeg,
        )
    except Exception as e:
        return failure_response(
            error_code="INTERNAL_ERROR",
            message=str(e),
            hint="Check server logs for details.",
        )
