import base64
import io
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image


def _result_base() -> Dict[str, Any]:
    return {
        "totalDrivingMinutes": 0,
        "totalStopMinutes": 0,
        "needsReviewMinutes": 0,
        "segments": [],
    }


def make_debug_image_base64(image_bytes: bytes) -> str:
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")

        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 255, 0), 6)
        cv2.putText(
            img,
            "OK",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 255, 0),
            6,
            cv2.LINE_AA,
        )

        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ValueError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        return base64.b64encode(png_bytes).decode("ascii")


def analyze_image(
    *,
    image_bytes: bytes,
    chart_type: Optional[str] = None,
    midnight_offset_deg: Optional[float] = None,
) -> Dict[str, Any]:
    res = _result_base()
    res["debugImageBase64"] = make_debug_image_base64(image_bytes)
    res["meta"] = {
        "chartType": chart_type,
        "midnightOffsetDeg": midnight_offset_deg,
    }
    return res


def failure_response(*, error_code: str, message: str, hint: str) -> Dict[str, Any]:
    res = _result_base()
    res["errorCode"] = error_code
    res["message"] = message
    res["hint"] = hint
    return res
