import base64
from typing import Any, Dict, Optional


def _result_base() -> Dict[str, Any]:
    return {
        "totalDrivingMinutes": 0,
        "totalStopMinutes": 0,
        "needsReviewMinutes": 0,
        "segments": [],
    }


def make_debug_image_base64(image_bytes: bytes) -> str:
    _ = image_bytes
    return base64.b64encode(b"").decode("ascii")


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
