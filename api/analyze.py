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


def detect_circle_hough(bgr: np.ndarray) -> Optional[Dict[str, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape[:2]
    min_dim = min(h, w)
    min_radius = int(min_dim * 0.30)
    max_radius = int(min_dim * 0.49)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim / 2,
        param1=120,
        param2=40,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None or circles.size == 0:
        return None

    best = max(circles[0], key=lambda c: c[2])
    cx, cy, r = [int(round(v)) for v in best]
    return {"cx": cx, "cy": cy, "r": r}


def _draw_center_marker(bgr: np.ndarray, cx: int, cy: int) -> None:
    cv2.drawMarker(
        bgr,
        (cx, cy),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=40,
        thickness=4,
        line_type=cv2.LINE_AA,
    )


def make_debug_image_base64(image_bytes: bytes) -> tuple[str, Optional[Dict[str, int]]]:
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

        circle = detect_circle_hough(img)
        if circle is not None:
            cv2.circle(
                img,
                (circle["cx"], circle["cy"]),
                circle["r"],
                (0, 255, 0),
                6,
                lineType=cv2.LINE_AA,
            )
            _draw_center_marker(img, circle["cx"], circle["cy"])

        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ValueError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii"), circle
    except Exception:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        return base64.b64encode(png_bytes).decode("ascii"), None


def analyze_image(
    *,
    image_bytes: bytes,
    chart_type: Optional[str] = None,
    midnight_offset_deg: Optional[float] = None,
) -> Dict[str, Any]:
    res = _result_base()
    debug_b64, circle = make_debug_image_base64(image_bytes)
    res["debugImageBase64"] = debug_b64
    res["meta"] = {
        "chartType": chart_type,
        "midnightOffsetDeg": midnight_offset_deg,
    }

    if circle is None:
        res["errorCode"] = "CIRCLE_NOT_FOUND"
        res["message"] = "Circle not found by HoughCircles."
        res["hint"] = "Try capturing the full chart with less reflection and better contrast."
    else:
        res["meta"]["circle"] = circle

    return res


def failure_response(*, error_code: str, message: str, hint: str) -> Dict[str, Any]:
    res = _result_base()
    res["errorCode"] = error_code
    res["message"] = message
    res["hint"] = hint
    return res
