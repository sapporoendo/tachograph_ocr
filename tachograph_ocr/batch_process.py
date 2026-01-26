import argparse
import html
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

import core_processor


IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


DEFAULT_THRESHOLDS = [15]
DEFAULT_CROSS_BAND_HALF_WIDTH = 3
DEFAULT_SMALL_CC_AREA_MAX = 20
DEFAULT_CROSS_WEIGHT = 4.0
DEFAULT_SMALLCC_WEIGHT = 1.0


@dataclass
class RunResult:
    image_name: str
    image_path: str
    threshold: int
    outdir: str
    ok: bool
    paths: Dict[str, str]
    metrics: Dict[str, float]
    error: Optional[str]


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _score_diff_clean(
    diff_clean_path: str,
    *,
    center_xy: Optional[Tuple[float, float]],
    cross_band_half_width: int,
    small_cc_area_max: int,
    cross_weight: float,
    smallcc_weight: float,
) -> Dict[str, float]:
    bw = cv2.imread(diff_clean_path, cv2.IMREAD_GRAYSCALE)
    if bw is None:
        raise ValueError(f"failed to read diff_clean: {diff_clean_path}")

    if bw.dtype != "uint8":
        bw = bw.astype("uint8")

    _, bw = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY)
    h, w = bw.shape[:2]

    if center_xy is None:
        cx = float(w) * 0.5
        cy = float(h) * 0.5
    else:
        cx = float(center_xy[0])
        cy = float(center_xy[1])

    ink_pixels = float(cv2.countNonZero(bw))

    x0 = _clamp_int(int(round(cx - float(cross_band_half_width))), 0, w - 1)
    x1 = _clamp_int(int(round(cx + float(cross_band_half_width))), 0, w - 1)
    y0 = _clamp_int(int(round(cy - float(cross_band_half_width))), 0, h - 1)
    y1 = _clamp_int(int(round(cy + float(cross_band_half_width))), 0, h - 1)

    # include both ends
    v_strip = bw[:, x0 : (x1 + 1)]
    h_strip = bw[y0 : (y1 + 1), :]
    inter = bw[y0 : (y1 + 1), x0 : (x1 + 1)]
    cross_penalty = float(cv2.countNonZero(v_strip) + cv2.countNonZero(h_strip) - cv2.countNonZero(inter))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    small_area_sum = 0.0
    for i in range(1, int(num)):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < float(small_cc_area_max):
            small_area_sum += area

    score = float(ink_pixels - float(cross_weight) * cross_penalty - float(smallcc_weight) * small_area_sum)

    return {
        "ink_pixels": float(ink_pixels),
        "cross_penalty": float(cross_penalty),
        "small_cc_penalty": float(small_area_sum),
        "score": float(score),
    }


def _iter_images(input_dir: Path) -> List[Path]:
    items: List[Path] = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {e.lower() for e in IMG_EXTS}:
            continue
        items.append(p)
    return items


def _pick_template(input_dir: Path, template_arg: Optional[str]) -> Path:
    if template_arg:
        p = Path(template_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"template not found: {p}")

    p1 = input_dir / "IMG_1035.jpg"
    if p1.exists():
        return p1

    p2 = input_dir / "template.jpg"
    if p2.exists():
        return p2

    raise FileNotFoundError("template not found. Put IMG_1035.jpg or template.jpg under input/ or pass --template")


def _safe_relpath(path: str, start: str) -> str:
    try:
        return os.path.relpath(path, start)
    except Exception:
        return path


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _build_report_html(
    *,
    out_root: Path,
    template_path: Path,
    thresholds: Sequence[int],
    results: List[RunResult],
    best_global_threshold: Optional[int],
    best_by_image: Dict[str, int],
    score_sum_by_threshold: Dict[int, float],
) -> str:
    rows: List[str] = []

    by_image: Dict[str, List[RunResult]] = {}
    for r in results:
        by_image.setdefault(r.image_name, []).append(r)

    for img_name in sorted(by_image.keys()):
        rr = sorted(by_image[img_name], key=lambda x: int(x.threshold))
        img_path = rr[0].image_path if rr else ""

        cells: List[str] = []
        for th in thresholds:
            match = next((x for x in rr if int(x.threshold) == int(th)), None)
            if match is None:
                cells.append("<td class='missing'>-</td>")
                continue

            is_best_img = int(best_by_image.get(img_name, -999999)) == int(th)
            is_best_global = best_global_threshold is not None and int(best_global_threshold) == int(th)
            td_cls = ""
            if is_best_img and is_best_global:
                td_cls = " class='best best-both'"
            elif is_best_img:
                td_cls = " class='best best-image'"
            elif is_best_global:
                td_cls = " class='best best-global'"

            if match.ok:
                reg = match.paths.get("debug_target_registered") or ""
                diff = match.paths.get("diff_clean") or ""
                reg_rel = html.escape(_safe_relpath(reg, str(out_root)))
                diff_rel = html.escape(_safe_relpath(diff, str(out_root)))

                m = match.metrics or {}
                ink = html.escape(str(int(round(float(m.get("ink_pixels", 0.0))))))
                cross = html.escape(str(int(round(float(m.get("cross_penalty", 0.0))))))
                small = html.escape(str(int(round(float(m.get("small_cc_penalty", 0.0))))))
                score = html.escape(str(round(float(m.get("score", 0.0)), 3)))
                cells.append(
                    """
<td{td_cls}>
  <div class='cell'>
    <div class='title'>thr={th}</div>
    <div class='metrics'>
      <div>ink: {ink}</div>
      <div>cross: {cross}</div>
      <div>small: {small}</div>
      <div><b>score: {score}</b></div>
    </div>
    <div class='imgs'>
      <a href='{reg_rel}' target='_blank'><img src='{reg_rel}'></a>
      <a href='{diff_rel}' target='_blank'><img src='{diff_rel}'></a>
    </div>
  </div>
</td>
""".format(
                        td_cls=td_cls,
                        th=int(th),
                        ink=ink,
                        cross=cross,
                        small=small,
                        score=score,
                        reg_rel=reg_rel,
                        diff_rel=diff_rel,
                    )
                )
            else:
                err = html.escape(match.error or "error")
                cells.append(
                    """
<td class='error'{td_cls}>
  <div class='cell'>
    <div class='title'>thr={th}</div>
    <pre>{err}</pre>
  </div>
</td>
""".format(
                        td_cls=td_cls,
                        th=int(th),
                        err=err,
                    )
                )

        rows.append(
            """
<tr>
  <th>
    <div class='rowhead'>
      <div class='name'>{name}</div>
      <div class='path'>{path}</div>
    </div>
  </th>
  {cells}
</tr>
""".format(
                name=html.escape(img_name),
                path=html.escape(img_path),
                cells="\n".join(cells),
            )
        )

    th_cols_parts: List[str] = []
    for t in thresholds:
        is_best_global = best_global_threshold is not None and int(best_global_threshold) == int(t)
        cls = " class='best-col'" if is_best_global else ""
        score_sum = float(score_sum_by_threshold.get(int(t), 0.0))
        th_cols_parts.append(f"<th{cls}>thr={int(t)}<div class='colsum'>sum: {round(score_sum, 1)}</div></th>")
    th_cols = "".join(th_cols_parts)

    best_line = "-" if best_global_threshold is None else str(int(best_global_threshold))

    return """<!doctype html>
<html lang='ja'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>tachograph batch report</title>
<style>
  body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans JP',sans-serif; margin: 16px; }}
  .meta {{ margin-bottom: 12px; color: #444; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; vertical-align: top; }}
  th {{ background: #fafafa; position: sticky; left: 0; z-index: 2; }}
  thead th {{ top: 0; position: sticky; z-index: 3; }}
  .rowhead {{ min-width: 240px; padding: 10px; }}
  .rowhead .name {{ font-weight: 700; }}
  .rowhead .path {{ font-size: 12px; color: #666; word-break: break-all; }}
  .cell {{ padding: 10px; }}
  .title {{ font-weight: 700; margin-bottom: 6px; }}
  .metrics {{ font-size: 12px; color: #333; margin-bottom: 8px; }}
  .imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  img {{ width: 100%; height: auto; max-width: 360px; background: #000; }}
  td.error {{ background: #fff3f3; }}
  td.missing {{ color: #999; text-align: center; padding: 18px; }}
  td.best {{ outline: 3px solid #2d7ff9; outline-offset: -3px; }}
  td.best-global {{ outline-color: #f59e0b; }}
  td.best-image {{ outline-color: #2d7ff9; }}
  td.best-both {{ outline-color: #10b981; }}
  th.best-col {{ background: #fff7ed; }}
  .colsum {{ font-size: 11px; color: #666; margin-top: 4px; }}
  pre {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; }}
</style>
</head>
<body>
  <div class='meta'>
    <div><b>template</b>: {template}</div>
    <div><b>thresholds</b>: {thresholds}</div>
    <div><b>おすすめthreshold</b>: {best_line}</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>image</th>
        {th_cols}
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
""".format(
        template=html.escape(str(template_path)),
        thresholds=html.escape(", ".join([str(int(t)) for t in thresholds])
        ),
        best_line=html.escape(best_line),
        th_cols=th_cols,
        rows="\n".join(rows),
    )


def _copy_or_link(src: str, dst: str) -> None:
    # Keep it simple: just copy for portability.
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, img)


def _sanitize_stem(stem: str) -> str:
    s = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in stem])
    return s[:120] if len(s) > 120 else s


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input")
    ap.add_argument("--output", default="output")
    ap.add_argument("--template", default="")
    ap.add_argument("--thresholds", default=",".join([str(int(t)) for t in DEFAULT_THRESHOLDS]))
    ap.add_argument("--include-otsu", action="store_true")
    ap.add_argument("--cross-band", type=int, default=int(DEFAULT_CROSS_BAND_HALF_WIDTH))
    ap.add_argument("--small-cc-max-area", type=int, default=int(DEFAULT_SMALL_CC_AREA_MAX))
    ap.add_argument("--cross-weight", type=float, default=float(DEFAULT_CROSS_WEIGHT))
    ap.add_argument("--smallcc-weight", type=float, default=float(DEFAULT_SMALLCC_WEIGHT))
    args = ap.parse_args(list(argv) if argv is not None else None)

    input_dir = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"input dir not found: {input_dir}", file=sys.stderr)
        return 2

    template_path = _pick_template(input_dir, args.template or None)

    thresholds: List[int] = []
    for tok in str(args.thresholds).split(","):
        tok = tok.strip()
        if not tok:
            continue
        thresholds.append(int(tok))
    if len(thresholds) == 0:
        thresholds = list(DEFAULT_THRESHOLDS)

    if bool(args.include_otsu) and 0 not in thresholds:
        thresholds = [0] + list(thresholds)

    thresholds = sorted(list(dict.fromkeys([int(t) for t in thresholds])))

    images = _iter_images(input_dir)
    images = [p for p in images if p.resolve() != template_path.resolve()]
    if len(images) == 0:
        print("no images found under input/", file=sys.stderr)
        return 2

    results: List[RunResult] = []

    for img_path in images:
        stem = _sanitize_stem(img_path.stem)
        for th in thresholds:
            run_out = out_root / stem / f"thr_{int(th)}"
            try:
                r = core_processor.process(
                    template_path=str(template_path),
                    target_path=str(img_path),
                    outdir=str(run_out),
                    threshold=int(th),
                )
                ok = bool(r.get("ok"))
                paths = (r.get("paths") or {}) if isinstance(r, dict) else {}

                metrics: Dict[str, float] = {}
                if ok and isinstance(paths, dict) and paths.get("diff_clean"):
                    center_xy = None
                    regp = r.get("registration") if isinstance(r, dict) else None
                    if isinstance(regp, dict) and isinstance(regp.get("template_center"), list) and len(regp.get("template_center")) >= 2:
                        center_xy = (float(regp["template_center"][0]), float(regp["template_center"][1]))
                    metrics = _score_diff_clean(
                        str(paths["diff_clean"]),
                        center_xy=center_xy,
                        cross_band_half_width=int(args.cross_band),
                        small_cc_area_max=int(args.small_cc_max_area),
                        cross_weight=float(args.cross_weight),
                        smallcc_weight=float(args.smallcc_weight),
                    )

                results.append(
                    RunResult(
                        image_name=stem,
                        image_path=str(img_path),
                        threshold=int(th),
                        outdir=str(run_out),
                        ok=ok,
                        paths={k: str(v) for k, v in paths.items()} if isinstance(paths, dict) else {},
                        metrics={k: float(v) for k, v in metrics.items()} if isinstance(metrics, dict) else {},
                        error=None,
                    )
                )
            except Exception as e:
                tb = traceback.format_exc(limit=12)
                results.append(
                    RunResult(
                        image_name=stem,
                        image_path=str(img_path),
                        threshold=int(th),
                        outdir=str(run_out),
                        ok=False,
                        paths={},
                        metrics={},
                        error=f"{type(e).__name__}: {e}\n{tb}",
                    )
                )
                continue

    # pick best per image and globally by score
    best_by_image: Dict[str, int] = {}
    score_sum_by_threshold: Dict[int, float] = {int(t): 0.0 for t in thresholds}

    for r in results:
        if not r.ok:
            continue
        score_sum_by_threshold[int(r.threshold)] = float(score_sum_by_threshold.get(int(r.threshold), 0.0)) + float(r.metrics.get("score", 0.0))

    for img_name in sorted({r.image_name for r in results}):
        candidates = [
            r
            for r in results
            if r.image_name == img_name and r.ok and isinstance(r.metrics, dict) and ("score" in r.metrics)
        ]
        if len(candidates) == 0:
            continue
        candidates.sort(
            key=lambda x: (
                float(x.metrics.get("score", -1e18)),
                float(x.metrics.get("ink_pixels", -1e18)),
                -int(x.threshold),
            ),
            reverse=True,
        )
        best_by_image[img_name] = int(candidates[0].threshold)

    best_global_threshold: Optional[int] = None
    if len(score_sum_by_threshold) > 0:
        best_global_threshold = sorted(
            score_sum_by_threshold.items(),
            key=lambda kv: (float(kv[1]), -int(kv[0])),
            reverse=True,
        )[0][0]

    report_html = _build_report_html(
        out_root=out_root,
        template_path=template_path,
        thresholds=thresholds,
        results=results,
        best_global_threshold=best_global_threshold,
        best_by_image=best_by_image,
        score_sum_by_threshold=score_sum_by_threshold,
    )
    report_path = out_root / "report.html"
    _write_text(report_path, report_html)

    summary = {
        "template": str(template_path),
        "thresholds": [int(t) for t in thresholds],
        "best_global_threshold": int(best_global_threshold) if best_global_threshold is not None else None,
        "best_by_image": {str(k): int(v) for k, v in best_by_image.items()},
        "score_sum_by_threshold": {str(int(k)): float(v) for k, v in score_sum_by_threshold.items()},
        "scoring": {
            "cross_band_half_width": int(args.cross_band),
            "small_cc_area_max": int(args.small_cc_max_area),
            "cross_weight": float(args.cross_weight),
            "smallcc_weight": float(args.smallcc_weight),
        },
        "images": [str(p) for p in images],
        "report": str(report_path),
        "results": [
            {
                "image": r.image_name,
                "path": r.image_path,
                "threshold": int(r.threshold),
                "ok": bool(r.ok),
                "outdir": r.outdir,
                "metrics": {str(k): float(v) for k, v in (r.metrics or {}).items()},
                "error": r.error,
            }
            for r in results
        ],
    }

    summary_path = out_root / "batch_summary.json"
    _write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"report: {report_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
