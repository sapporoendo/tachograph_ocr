import base64
import binascii
import os
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .analyze import analyze_image, failure_response
from .history_db import connect, get_record, init_db, insert_record, list_records, save_debug_png_bytes, save_image_bytes, set_checked

app = FastAPI(title="tachograph_ocr api")


@app.on_event("startup")
def _startup() -> None:
    conn = connect()
    try:
        init_db(conn)
    finally:
        conn.close()


def _parse_cors_allow_origins() -> list[str]:
    v = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    if not v:
        return ["*"]
    if v == "*":
        return ["*"]
    parts = [p.strip() for p in v.split(",")]
    return [p for p in parts if p]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_allow_origins(),
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


@app.post("/records")
async def create_record(
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    driverName: Optional[str] = Form(default=None),
    vehicleNo: Optional[str] = Form(default=None),
    distanceKm: Optional[float] = Form(default=None),
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

    raw = await up.read()
    if not raw:
        return failure_response(
            error_code="EMPTY_FILE",
            message="Uploaded file is empty.",
            hint="Please re-upload the image.",
        )

    result = analyze_image(
        image_bytes=raw,
        chart_type=chartType,
        midnight_offset_deg=midnightOffsetDeg,
    )

    analysis_to_store: Dict[str, Any] = result if isinstance(result, dict) else {}
    if isinstance(analysis_to_store, dict) and isinstance(analysis_to_store.get("debugImageBase64"), str):
        analysis_to_store = dict(analysis_to_store)
        analysis_to_store.pop("debugImageBase64", None)

    data_dir = os.getenv("TACHO_DATA_DIR")
    record_id, image_path = save_image_bytes(
        data_dir=data_dir or os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data"),
        filename_hint=up.filename or "tachograph.jpg",
        image_bytes=raw,
    )

    debug_image_path = None
    dbg_b64 = result.get("debugImageBase64") if isinstance(result, dict) else None
    if isinstance(dbg_b64, str) and dbg_b64:
        try:
            dbg_bytes = base64.b64decode(dbg_b64)
            debug_image_path = save_debug_png_bytes(
                data_dir=data_dir or os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data"),
                record_id=record_id,
                png_bytes=dbg_bytes,
            )
        except (binascii.Error, ValueError):
            debug_image_path = None

    conn = connect(data_dir)
    try:
        init_db(conn)
        insert_record(
            conn=conn,
            record_id=record_id,
            driver_name=driverName,
            vehicle_no=vehicleNo,
            distance_km=distanceKm,
            chart_type=chartType,
            midnight_offset_deg=midnightOffsetDeg,
            image_path=image_path,
            debug_image_path=debug_image_path,
            analysis_json=analysis_to_store,
        )
    finally:
        conn.close()

    if isinstance(result, dict):
        result["recordId"] = record_id
    return result


@app.get("/records")
def get_records(
    limit: int = 100,
    offset: int = 0,
    checked: Optional[bool] = None,
) -> dict:
    conn = connect()
    try:
        init_db(conn)
        rows = list_records(conn=conn, limit=limit, offset=offset, checked=checked)
    finally:
        conn.close()
    return {"records": rows}


@app.get("/records/{record_id}")
def get_record_detail(record_id: str) -> dict:
    conn = connect()
    try:
        init_db(conn)
        rec = get_record(conn=conn, record_id=record_id)
    finally:
        conn.close()

    if rec is None:
        raise HTTPException(status_code=404, detail="record not found")
    return rec


@app.get("/records/{record_id}/image")
def get_record_image(record_id: str) -> FileResponse:
    conn = connect()
    try:
        init_db(conn)
        rec = get_record(conn=conn, record_id=record_id)
    finally:
        conn.close()

    if rec is None:
        raise HTTPException(status_code=404, detail="record not found")
    p = rec.get("imagePath")
    if not isinstance(p, str) or not p or not os.path.exists(p):
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(p)


@app.get("/records/{record_id}/debug.png")
def get_record_debug_image(record_id: str) -> FileResponse:
    conn = connect()
    try:
        init_db(conn)
        rec = get_record(conn=conn, record_id=record_id)
    finally:
        conn.close()

    if rec is None:
        raise HTTPException(status_code=404, detail="record not found")
    p = rec.get("debugImagePath")
    if not isinstance(p, str) or not p or not os.path.exists(p):
        raise HTTPException(status_code=404, detail="debug image not found")
    return FileResponse(p, media_type="image/png")


@app.patch("/records/{record_id}/checked")
def update_record_checked(record_id: str, body: Dict[str, Any] = Body(...)) -> dict:
    checked = body.get("checked")
    if not isinstance(checked, bool):
        raise HTTPException(status_code=400, detail="checked must be boolean")

    conn = connect()
    try:
        init_db(conn)
        ok = set_checked(conn=conn, record_id=record_id, checked=checked)
    finally:
        conn.close()

    if not ok:
        raise HTTPException(status_code=404, detail="record not found")
    return {"ok": True, "id": record_id, "checked": checked}
