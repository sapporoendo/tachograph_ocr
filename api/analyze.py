import base64
import io
import math
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


def _draw_compass(bgr: np.ndarray, cx: int, cy: int, r: int) -> None:
    L = max(20, int(r * 0.85))
    directions = [
        (0, (255, 255, 255), "0"),
        (90, (255, 255, 255), "90"),
        (180, (255, 255, 255), "180"),
        (270, (255, 255, 255), "270"),
    ]

    for deg, color, label in directions:
        theta = math.radians(float(deg))
        x2 = int(round(cx + L * math.cos(theta)))
        y2 = int(round(cy - L * math.sin(theta)))
        cv2.line(bgr, (cx, cy), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            bgr,
            label,
            (x2 + 6, y2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def angle_to_time_24h(
    *,
    needle_angle_deg: float,
    midnight_offset_deg: float,
    clockwise_increases: bool = True,
) -> Dict[str, Any]:
    if clockwise_increases:
        rel = (needle_angle_deg - midnight_offset_deg) % 360.0
        formula = "rel=(needleAngleDeg - midnightOffsetDeg)%360; minute=int(rel/360*1440)"
    else:
        rel = (midnight_offset_deg - needle_angle_deg) % 360.0
        formula = "rel=(midnightOffsetDeg - needleAngleDeg)%360; minute=int(rel/360*1440)"

    minute = int((rel / 360.0) * 1440.0) % 1440
    hh = minute // 60
    mm = minute % 60
    hhmm = f"{hh:02d}:{mm:02d}"
    return {
        "minuteOfDay": minute,
        "timeHHMM": hhmm,
        "relAngleDeg": rel,
        "formula": formula,
    }


def _point_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1

    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(px - x1, py - y1)

    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(px - x2, py - y2)

    t = c1 / c2
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return math.hypot(px - proj_x, py - proj_y)


def detect_needle_polar_score(
    bgr: np.ndarray,
    *,
    cx: int,
    cy: int,
    r: int,
    force_angle: bool = True,
) -> Optional[Dict[str, Any]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)

    canny1 = 30
    canny2 = 100
    edges = cv2.Canny(gray, canny1, canny2)

    R = max(10, int(r * 0.90))
    polar = cv2.warpPolar(
        edges,
        (360, R),
        (float(cx), float(cy)),
        float(R),
        cv2.WARP_POLAR_LINEAR,
    )

    inner = int(0.15 * R)
    outer = int(0.65 * R)
    inner = max(0, min(inner, R - 1))
    outer = max(inner + 1, min(outer, R))

    band = polar[inner:outer, :]
    scores = band.sum(axis=0).astype(np.float32)

    win = 9
    kernel = np.ones(win, dtype=np.float32) / float(win)
    padded = np.r_[scores[-(win // 2) :], scores, scores[: (win // 2)]]
    smooth = np.convolve(padded, kernel, mode="valid")

    best_idx = int(np.argmax(smooth))
    order = np.argsort(smooth)[::-1]
    topk = [int(order[i]) for i in range(min(3, len(order)))]
    best_score = float(smooth[best_idx])
    second_score = float(smooth[topk[1]]) if len(topk) > 1 else 0.0
    confidence = float(best_score / (second_score + 1e-6))

    top3 = [{"angleIdx": int(a), "score": float(smooth[int(a)])} for a in topk]

    if (not force_angle) and best_score <= 0.0:
        return None

    return {
        "angleDeg": float(best_idx),
        "bestScore": best_score,
        "topAngles": topk,
        "top3": top3,
        "confidence": confidence,
        "inner": inner,
        "outer": outer,
        "canny1": canny1,
        "canny2": canny2,
        "win": win,
    }


def detect_needle_hough_linesp(
    bgr: np.ndarray,
    *,
    cx: int,
    cy: int,
    r: int,
    debug_overlay_bgr: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    mask = np.zeros_like(gray)
    roi_r = max(10, int(r * 0.90))
    cv2.circle(mask, (cx, cy), roi_r, 255, thickness=-1)
    roi = cv2.bitwise_and(gray, mask)

    edges = cv2.Canny(roi, 30, 100)

    min_line_len = max(25, int(r * 0.30))
    max_line_gap = max(12, int(r * 0.12))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    if lines is None or len(lines) == 0:
        return None

    center_dist_thresh = max(18.0, min(60.0, r * 0.12))
    endpoint_near_center_thresh = max(30.0, min(90.0, r * 0.22))

    best = None
    best_len = 0.0

    for l in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in l]

        if debug_overlay_bgr is not None:
            cv2.line(
                debug_overlay_bgr,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )

        d = _point_segment_distance(float(cx), float(cy), x1, y1, x2, y2)
        if d > center_dist_thresh:
            continue

        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        if min(d1, d2) > endpoint_near_center_thresh:
            continue

        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len <= best_len:
            continue

        best = (int(x1), int(y1), int(x2), int(y2))
        best_len = seg_len

    if best is None:
        return None

    x1, y1, x2, y2 = best
    d1 = math.hypot(x1 - cx, y1 - cy)
    d2 = math.hypot(x2 - cx, y2 - cy)
    if d1 >= d2:
        tip_x, tip_y = x1, y1
        base_x, base_y = x2, y2
    else:
        tip_x, tip_y = x2, y2
        base_x, base_y = x1, y1

    dx = float(tip_x - cx)
    dy = float(cy - tip_y)
    angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

    return {
        "x1": base_x,
        "y1": base_y,
        "x2": tip_x,
        "y2": tip_y,
        "angleDeg": angle,
    }


def make_debug_image_base64(
    image_bytes: bytes,
    midnight_offset_deg: Optional[float] = None,
) -> tuple[str, Optional[Dict[str, int]], Optional[float], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img0 is None:
            raise ValueError("cv2.imdecode returned None")

        img = img0.copy()

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

        circle = detect_circle_hough(img0)
        needle_angle: Optional[float] = None
        polar_info: Optional[Dict[str, Any]] = None
        time_info: Optional[Dict[str, Any]] = None
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
            _draw_compass(img, circle["cx"], circle["cy"], circle["r"])

            needle = detect_needle_polar_score(
                img0,
                cx=circle["cx"],
                cy=circle["cy"],
                r=circle["r"],
                force_angle=True,
            )
            if needle is not None:
                polar_info = needle
                needle_angle = float(needle["angleDeg"])

                if midnight_offset_deg is not None:
                    time_info = angle_to_time_24h(
                        needle_angle_deg=needle_angle,
                        midnight_offset_deg=float(midnight_offset_deg),
                        clockwise_increases=True,
                    )
                    cv2.putText(
                        img,
                        f"t={time_info['timeHHMM']}",
                        (w - 260, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (255, 255, 255),
                        4,
                        cv2.LINE_AA,
                    )

                top_angles = needle.get("topAngles") or []
                for idx, a in enumerate(top_angles[:3]):
                    theta = math.radians(float(a))
                    L = int(circle["r"] * 0.80)
                    x2 = int(round(circle["cx"] + L * math.cos(theta)))
                    y2 = int(round(circle["cy"] - L * math.sin(theta)))

                    if idx == 0:
                        color = (0, 0, 255)
                        thickness = 8
                    elif idx == 1:
                        color = (0, 128, 255)
                        thickness = 4
                    else:
                        color = (255, 0, 255)
                        thickness = 3

                    cv2.line(
                        img,
                        (circle["cx"], circle["cy"]),
                        (x2, y2),
                        color,
                        thickness,
                        lineType=cv2.LINE_AA,
                    )

                print(
                    "needle_polar:",
                    "peak=", int(needle.get("angleDeg", 0.0)),
                    "score_max=", round(float(needle.get("bestScore", 0.0)), 1),
                    "top3=", needle.get("topAngles"),
                    "inner=", int(needle.get("inner", 0)),
                    "outer=", int(needle.get("outer", 0)),
                    "conf=", round(float(needle.get("confidence", 0.0)), 3),
                )

        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ValueError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii"), circle, needle_angle, polar_info, time_info
    except Exception:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        return base64.b64encode(png_bytes).decode("ascii"), None, None, None, None


def analyze_image(
    *,
    image_bytes: bytes,
    chart_type: Optional[str] = None,
    midnight_offset_deg: Optional[float] = None,
) -> Dict[str, Any]:
    res = _result_base()
    debug_b64, circle, needle_angle, polar_info, time_info = make_debug_image_base64(
        image_bytes,
        midnight_offset_deg=midnight_offset_deg,
    )
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

        if polar_info is not None:
            peak = int(polar_info.get("angleDeg", 0.0))
            score_max = float(polar_info.get("bestScore", 0.0))
            top_angles = polar_info.get("topAngles")
            inner = int(polar_info.get("inner", 0))
            outer = int(polar_info.get("outer", 0))
            conf = float(polar_info.get("confidence", 0.0))

            res["meta"]["polarLog"] = (
                f"needle_polar: peak={peak} score_max={score_max:.1f} top3={top_angles} "
                f"inner={inner} outer={outer} conf={conf:.3f} angleDeg={peak}"
            )
            res["meta"]["polarTop3"] = polar_info.get("top3")
            res["meta"]["polarParams"] = {
                "inner": inner,
                "outer": outer,
                "canny1": int(polar_info.get("canny1", 0)),
                "canny2": int(polar_info.get("canny2", 0)),
                "win": int(polar_info.get("win", 0)),
            }

        if time_info is not None:
            res["meta"]["needleMinuteOfDay"] = int(time_info.get("minuteOfDay", 0))
            res["meta"]["needleTimeHHMM"] = time_info.get("timeHHMM")
            res["meta"]["angleToMinuteFormula"] = time_info.get("formula")

        if needle_angle is None:
            res["errorCode"] = "NEEDLE_NOT_FOUND"
            res["message"] = "Needle line not detected."
            res["hint"] = "Try clearer photo, avoid glare, and capture the full chart."
        else:
            res["meta"]["needleAngleDeg"] = needle_angle
            res["message"] = "Day2: needleAngle->time implemented; segments/driving-stop estimation is next."

    return res


def failure_response(*, error_code: str, message: str, hint: str) -> Dict[str, Any]:
    res = _result_base()
    res["errorCode"] = error_code
    res["message"] = message
    res["hint"] = hint
    return res
