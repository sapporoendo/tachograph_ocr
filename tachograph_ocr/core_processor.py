import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LineModel:
    kind: str
    a: float
    b: float


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _imread_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to read image: {path}")
    return img


def _center_roi_mask(shape_hw: Tuple[int, int], *, roi_ratio: float = 0.80) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    cx, cy = w * 0.5, h * 0.5
    half_w = (w * float(roi_ratio)) * 0.5
    half_h = (h * float(roi_ratio)) * 0.5
    x0 = int(max(0, round(cx - half_w)))
    x1 = int(min(w, round(cx + half_w)))
    y0 = int(max(0, round(cy - half_h)))
    y1 = int(min(h, round(cy + half_h)))
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def _apply_clahe_gray(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)


def _fit_line_from_points(points_xy: np.ndarray, *, prefer: str) -> Optional[LineModel]:
    if points_xy is None or points_xy.size == 0:
        return None
    pts = points_xy.astype(np.float64)
    xs = pts[:, 0]
    ys = pts[:, 1]

    if prefer == "horizontal":
        if xs.size < 2:
            return None
        a, b = np.polyfit(xs, ys, 1)
        return LineModel(kind="y= ax + b", a=float(a), b=float(b))

    if prefer == "vertical":
        if ys.size < 2:
            return None
        c, d = np.polyfit(ys, xs, 1)
        return LineModel(kind="x= cy + d", a=float(c), b=float(d))

    return None


def _line_intersection(vline: LineModel, hline: LineModel) -> Optional[Tuple[float, float]]:
    if vline.kind == "x= cy + d" and hline.kind == "y= ax + b":
        c, d = float(vline.a), float(vline.b)
        a, b = float(hline.a), float(hline.b)
        denom = 1.0 - (a * c)
        if abs(denom) < 1e-9:
            return None
        y = (a * d + b) / denom
        x = c * y + d
        return float(x), float(y)

    if vline.kind == "x= const" and hline.kind == "y= ax + b":
        x = float(vline.b)
        y = float(hline.a) * x + float(hline.b)
        return float(x), float(y)

    if vline.kind == "x= cy + d" and hline.kind == "y= const":
        y = float(hline.b)
        x = float(vline.a) * y + float(vline.b)
        return float(x), float(y)

    if vline.kind == "x= const" and hline.kind == "y= const":
        return float(vline.b), float(hline.b)

    return None


def detect_cross_center(
    bgr: np.ndarray,
    *,
    roi_ratio: float = 0.80,
    white_s_max: int = 60,
    white_v_min: int = 170,
) -> Dict[str, Any]:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    roi_mask = _center_roi_mask((h, w), roi_ratio=float(roi_ratio))
    white = cv2.inRange(hsv, (0, 0, int(white_v_min)), (179, int(white_s_max), 255))
    white[roi_mask == 0] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k, iterations=1)

    edges = cv2.Canny(white, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=int(0.25 * float(min(h, w))),
        maxLineGap=25,
    )

    if lines is None or len(lines) == 0:
        return {
            "ok": False,
            "center": (float(w) * 0.5, float(h) * 0.5),
            "vertical_line": None,
            "horizontal_line": None,
            "debug_bgr": bgr.copy(),
            "diagnostics": {"reason": "no_hough_lines"},
        }

    v_pts = []
    h_pts = []
    v_len = []
    h_len = []

    for ln in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [int(v) for v in ln]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < 20:
            continue
        ang = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
        if abs(ang - 90.0) <= 12.0:
            v_pts.append((x1, y1))
            v_pts.append((x2, y2))
            v_len.append(length)
        elif ang <= 12.0 or abs(ang - 180.0) <= 12.0:
            h_pts.append((x1, y1))
            h_pts.append((x2, y2))
            h_len.append(length)

    if len(v_pts) < 4 or len(h_pts) < 4:
        return {
            "ok": False,
            "center": (float(w) * 0.5, float(h) * 0.5),
            "vertical_line": None,
            "horizontal_line": None,
            "debug_bgr": bgr.copy(),
            "diagnostics": {"reason": "insufficient_vertical_or_horizontal", "v_pts": len(v_pts), "h_pts": len(h_pts)},
        }

    v_points = np.array(v_pts, dtype=np.float64)
    h_points = np.array(h_pts, dtype=np.float64)

    v_model = _fit_line_from_points(v_points, prefer="vertical")
    h_model = _fit_line_from_points(h_points, prefer="horizontal")
    if v_model is None or h_model is None:
        return {
            "ok": False,
            "center": (float(w) * 0.5, float(h) * 0.5),
            "vertical_line": None,
            "horizontal_line": None,
            "debug_bgr": bgr.copy(),
            "diagnostics": {"reason": "fit_failed"},
        }

    inter = _line_intersection(v_model, h_model)
    if inter is None:
        return {
            "ok": False,
            "center": (float(w) * 0.5, float(h) * 0.5),
            "vertical_line": None,
            "horizontal_line": None,
            "debug_bgr": bgr.copy(),
            "diagnostics": {"reason": "intersection_failed"},
        }

    cx, cy = inter

    dbg = bgr.copy()
    x0, x1 = 0, w - 1
    y0, y1 = 0, h - 1

    def _pt_on_h(x: int) -> Tuple[int, int]:
        yy = float(h_model.a) * float(x) + float(h_model.b)
        return int(x), int(round(yy))

    def _pt_on_v(y: int) -> Tuple[int, int]:
        xx = float(v_model.a) * float(y) + float(v_model.b)
        return int(round(xx)), int(y)

    p_h0 = _pt_on_h(x0)
    p_h1 = _pt_on_h(x1)
    p_v0 = _pt_on_v(y0)
    p_v1 = _pt_on_v(y1)

    cv2.line(dbg, p_h0, p_h1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(dbg, p_v0, p_v1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(dbg, (int(round(cx)), int(round(cy))), 8, (255, 0, 0), -1, lineType=cv2.LINE_AA)

    return {
        "ok": True,
        "center": (float(cx), float(cy)),
        "vertical_line": {"kind": v_model.kind, "c": float(v_model.a), "d": float(v_model.b)},
        "horizontal_line": {"kind": h_model.kind, "a": float(h_model.a), "b": float(h_model.b)},
        "debug_bgr": dbg,
        "diagnostics": {"roi_ratio": float(roi_ratio), "white_s_max": int(white_s_max), "white_v_min": int(white_v_min)},
    }


def detect_yellow_marker_angle(
    bgr: np.ndarray,
    center: Tuple[float, float],
    *,
    roi_ratio: float = 0.95,
    h_min: int = 15,
    h_max: int = 40,
    s_min: int = 80,
    v_min: int = 80,
    min_area: int = 200,
) -> Dict[str, Any]:
    h, w = bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])

    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (int(h_min), int(s_min), int(v_min)), (int(h_max), 255, 255))

        roi_mask = _center_roi_mask((h, w), roi_ratio=float(roi_ratio))
        mask[roi_mask == 0] = 0

        ys, xs = np.ogrid[:h, :w]
        r = float(min(h, w))
        r0 = 0.35 * r
        r1 = 0.65 * r
        dist2 = (xs.astype(np.float32) - cx) ** 2 + (ys.astype(np.float32) - cy) ** 2
        radial = (dist2 >= (r0 * r0)) & (dist2 <= (r1 * r1))
        mask[~radial] = 0

        y_cut = int(max(0, min(h, int(math.floor(cy)))))
        mask[y_cut:, :] = 0

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0.0
        for c in cnts or []:
            area = float(cv2.contourArea(c))
            if area < float(min_area):
                continue
            if area > best_area:
                best_area = area
                best = c

        dbg = bgr.copy()
        marker = None
        rot_deg = None

        if best is not None:
            m = cv2.moments(best)
            if abs(float(m.get("m00", 0.0))) > 1e-9:
                ux = float(m["m10"]) / float(m["m00"])
                uy = float(m["m01"]) / float(m["m00"])
                marker = (ux, uy)
                dx = ux - cx
                dy = uy - cy
                ang = math.degrees(math.atan2(dy, dx))
                rot_deg = float(-90.0 - ang)
                cv2.drawContours(dbg, [best], -1, (0, 255, 255), 2)
                cv2.circle(dbg, (int(round(ux)), int(round(uy))), 8, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        cv2.circle(dbg, (int(round(cx)), int(round(cy))), 6, (255, 0, 0), -1, lineType=cv2.LINE_AA)

        return {
            "ok": marker is not None and rot_deg is not None,
            "marker": marker,
            "angle_deg": rot_deg,
            "mask": mask,
            "debug_bgr": dbg,
            "diagnostics": {
                "roi_ratio": float(roi_ratio),
                "h_min": int(h_min),
                "h_max": int(h_max),
                "s_min": int(s_min),
                "v_min": int(v_min),
                "min_area": int(min_area),
                "upper_half_y_cut": int(y_cut),
            },
        }
    except Exception as e:
        dbg = bgr.copy()
        cv2.circle(dbg, (int(round(cx)), int(round(cy))), 6, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        empty = np.zeros((h, w), dtype=np.uint8)
        return {
            "ok": False,
            "marker": None,
            "angle_deg": 0.0,
            "mask": empty,
            "debug_bgr": dbg,
            "diagnostics": {
                "roi_ratio": float(roi_ratio),
                "h_min": int(h_min),
                "h_max": int(h_max),
                "s_min": int(s_min),
                "v_min": int(v_min),
                "min_area": int(min_area),
                "error": f"{type(e).__name__}: {e}",
            },
        }


def _warp_affine_keep_size(bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    return cv2.warpAffine(bgr, M, (int(w), int(h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def register_to_template(template_bgr: np.ndarray, target_bgr: np.ndarray) -> Dict[str, Any]:
    t_cross = detect_cross_center(template_bgr)
    g_cross = detect_cross_center(target_bgr)

    t_center = tuple(t_cross.get("center") or (template_bgr.shape[1] * 0.5, template_bgr.shape[0] * 0.5))
    g_center = tuple(g_cross.get("center") or (target_bgr.shape[1] * 0.5, target_bgr.shape[0] * 0.5))

    t_mark = detect_yellow_marker_angle(template_bgr, t_center)
    g_mark = detect_yellow_marker_angle(target_bgr, g_center)

    t_ang = float(t_mark.get("angle_deg") or 0.0)
    g_ang = float(g_mark.get("angle_deg") or 0.0)

    Mt = cv2.getRotationMatrix2D((float(t_center[0]), float(t_center[1])), float(t_ang), 1.0)
    Mg = cv2.getRotationMatrix2D((float(g_center[0]), float(g_center[1])), float(g_ang), 1.0)

    template_reg = _warp_affine_keep_size(template_bgr, Mt)

    dx = float(t_center[0]) - float(g_center[0])
    dy = float(t_center[1]) - float(g_center[1])
    Mg2 = Mg.copy()
    Mg2[0, 2] += dx
    Mg2[1, 2] += dy
    target_reg = _warp_affine_keep_size(target_bgr, Mg2)

    return {
        "template_registered": template_reg,
        "target_registered": target_reg,
        "params": {
            "template_center": [float(t_center[0]), float(t_center[1])],
            "target_center": [float(g_center[0]), float(g_center[1])],
            "template_angle_deg": float(t_ang),
            "target_angle_deg": float(g_ang),
            "matrix_template": Mt.tolist(),
            "matrix_target": Mg2.tolist(),
            "shift_dx": float(dx),
            "shift_dy": float(dy),
        },
        "debug": {
            "template_cross": t_cross,
            "target_cross": g_cross,
            "template_marker": t_mark,
            "target_marker": g_mark,
        },
    }


def subtract_with_clahe(
    template_registered_bgr: np.ndarray,
    target_registered_bgr: np.ndarray,
    *,
    threshold: int = 15,
) -> Dict[str, Any]:
    t_gray = _apply_clahe_gray(template_registered_bgr)
    g_gray = _apply_clahe_gray(target_registered_bgr)

    diff = cv2.absdiff(g_gray, t_gray)
    diff_raw = diff.copy()

    diff = cv2.GaussianBlur(diff, (3, 3), 0)

    if int(threshold) > 0:
        _, bw = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
        thr_used = int(threshold)
        thr_method = "fixed"
    else:
        thr_used, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_used = int(thr_used)
        thr_method = "otsu"

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    return {
        "diff_raw": diff_raw,
        "diff_clean": bw,
        "diagnostics": {"threshold_method": str(thr_method), "threshold_used": int(thr_used)},
    }


def process(
    *,
    template_path: str,
    target_path: str,
    outdir: str,
    threshold: int = 15,
) -> Dict[str, Any]:
    _ensure_dir(outdir)

    template_bgr = _imread_bgr(template_path)
    target_bgr = _imread_bgr(target_path)

    reg = register_to_template(template_bgr, target_bgr)
    template_reg = reg["template_registered"]
    target_reg = reg["target_registered"]

    t_cross_dbg = reg.get("debug", {}).get("template_cross", {}).get("debug_bgr")
    g_cross_dbg = reg.get("debug", {}).get("target_cross", {}).get("debug_bgr")

    debug_template_cross_path = os.path.join(outdir, "debug_template_cross.png")
    debug_target_cross_path = os.path.join(outdir, "debug_target_cross.png")
    debug_target_registered_path = os.path.join(outdir, "debug_target_registered.png")

    if isinstance(t_cross_dbg, np.ndarray):
        cv2.imwrite(debug_template_cross_path, t_cross_dbg)
    else:
        cv2.imwrite(debug_template_cross_path, template_bgr)

    if isinstance(g_cross_dbg, np.ndarray):
        cv2.imwrite(debug_target_cross_path, g_cross_dbg)
    else:
        cv2.imwrite(debug_target_cross_path, target_bgr)

    cv2.imwrite(debug_target_registered_path, target_reg)

    sub = subtract_with_clahe(template_reg, target_reg, threshold=int(threshold))
    diff_raw = sub["diff_raw"]
    diff_clean = sub["diff_clean"]

    diff_raw_path = os.path.join(outdir, "diff_raw.png")
    diff_clean_path = os.path.join(outdir, "diff_clean.png")
    cv2.imwrite(diff_raw_path, diff_raw)
    cv2.imwrite(diff_clean_path, diff_clean)

    result = {
        "ok": True,
        "paths": {
            "debug_template_cross": debug_template_cross_path,
            "debug_target_cross": debug_target_cross_path,
            "debug_target_registered": debug_target_registered_path,
            "diff_raw": diff_raw_path,
            "diff_clean": diff_clean_path,
        },
        "registration": reg.get("params"),
        "subtraction": sub.get("diagnostics"),
        "input": {
            "template_path": str(template_path),
            "target_path": str(target_path),
            "template_shape": [int(v) for v in template_bgr.shape],
            "target_shape": [int(v) for v in target_bgr.shape],
        },
        "debug": {
            "template_cross": reg.get("debug", {}).get("template_cross", {}).get("diagnostics"),
            "target_cross": reg.get("debug", {}).get("target_cross", {}).get("diagnostics"),
            "template_marker": reg.get("debug", {}).get("template_marker", {}).get("diagnostics"),
            "target_marker": reg.get("debug", {}).get("target_marker", {}).get("diagnostics"),
        },
    }

    meta_path = os.path.join(outdir, "result.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    result["paths"]["result_json"] = meta_path
    return result


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold", type=int, default=15)
    args = ap.parse_args()

    res = process(template_path=args.template, target_path=args.target, outdir=args.outdir, threshold=int(args.threshold))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    _main()
