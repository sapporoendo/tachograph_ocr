#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tacho-check [--restart-api] [--open-dir] [--base-url URL] [--out-root DIR] <image_or_dir>

Examples:
  tacho-check /path/to/DSCF2211.JPG
  tacho-check --restart-api /path/to/DSCF2211.JPG
  tacho-check --open-dir /path/to/images_dir

Notes:
- Output is saved under: ./output/ (or under --out-root DIR/output/)
- API base url resolution priority:
  1) --base-url
  2) .env API_BASE_URL
  3) env API_BASE_URL
  4) env ANALYZER_API_URL (strip /analyze)
  5) default (8000; or 8003 when --restart-api)
USAGE
}

RESTART_API=false
OPEN_DIR=false
BASE_URL=""
OUT_ROOT="${PWD}"
CHART_TYPE="24h"
MIDNIGHT_OFFSET_DEG="0"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart-api)
      RESTART_API=true
      shift
      ;;
    --open-dir)
      OPEN_DIR=true
      shift
      ;;
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="${2:-}"
      shift 2
      ;;
    --chart-type)
      CHART_TYPE="${2:-}"
      shift 2
      ;;
    --midnight-offset)
      MIDNIGHT_OFFSET_DEG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
POSITIONAL+=("$@")

if [[ ${#POSITIONAL[@]} -lt 1 ]]; then
  usage >&2
  exit 2
fi

INPUT_PATH="${POSITIONAL[0]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR_BASE="${OUT_ROOT%/}/output"
mkdir -p "${OUT_DIR_BASE}"

RUNS_DIR="${OUT_DIR_BASE}/runs"
mkdir -p "${RUNS_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

TMP_JSON="${RUN_DIR}/_response_tmp.json"

resolve_base_url() {
  local resolved="${BASE_URL}"
  local dotenv_path="${REPO_ROOT}/.env"

  if [[ -z "${resolved}" && -f "${dotenv_path}" ]]; then
    local env_base_url
    env_base_url=$(
      python3 - "${dotenv_path}" <<'PY'
import sys
path = sys.argv[1]
val = ""
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export "):].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        if k.strip() != "API_BASE_URL":
            continue
        v = v.strip().strip('"').strip("'")
        val = v
        break
print(val)
PY
    )
    if [[ -n "${env_base_url}" ]]; then
      resolved="${env_base_url}"
    fi
  fi

  if [[ -z "${resolved}" && -n "${API_BASE_URL:-}" ]]; then
    resolved="${API_BASE_URL}"
  fi

  if [[ -z "${resolved}" && -n "${ANALYZER_API_URL:-}" ]]; then
    resolved="${ANALYZER_API_URL%/analyze}"
  fi

  if [[ -z "${resolved}" ]]; then
    if [[ "${RESTART_API}" == "true" ]]; then
      resolved="http://127.0.0.1:8003"
    else
      resolved="http://127.0.0.1:8003"
    fi
  fi

  printf '%s' "${resolved}"
}

guess_repo_dir_for_restart() {
  if [[ -n "${TACHO_REPO_DIR:-}" && -d "${TACHO_REPO_DIR}/api" ]]; then
    printf '%s' "${TACHO_REPO_DIR}"
    return
  fi
  if [[ -d "./api" && -f "./api/main.py" ]]; then
    pwd
    return
  fi
  if [[ -d "${REPO_ROOT}/api" && -f "${REPO_ROOT}/api/main.py" ]]; then
    printf '%s' "${REPO_ROOT}"
    return
  fi
  printf '%s' ""
}

restart_api_8003() {
  local port=8003
  local repo_dir
  repo_dir="$(guess_repo_dir_for_restart)"
  if [[ -z "${repo_dir}" ]]; then
    echo "ERROR: could not locate repo dir for restart. Set TACHO_REPO_DIR or run from repo root." >&2
    return 1
  fi

  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:${port} 2>/dev/null || true)"
  fi

  if [[ -n "${pids}" ]]; then
    kill ${pids} >/dev/null 2>&1 || true
    sleep 0.5
    if command -v lsof >/dev/null 2>&1; then
      pids="$(lsof -ti tcp:${port} 2>/dev/null || true)"
    else
      pids=""
    fi
    if [[ -n "${pids}" ]]; then
      kill -9 ${pids} >/dev/null 2>&1 || true
    fi
  fi

  (
    cd "${repo_dir}" || exit 1
    nohup python3 -m uvicorn api.main:app --host 127.0.0.1 --port ${port} >"${OUT_DIR_BASE}/uvicorn_${port}.log" 2>&1 &
    echo $! >"${OUT_DIR_BASE}/uvicorn_${port}.pid"
  )
}

collect_images() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    shopt -s nullglob
    local -a imgs=()
    local ext
    for ext in jpg jpeg png JPG JPEG PNG; do
      local f
      for f in "${path}"/*."${ext}"; do
        imgs+=("${f}")
      done
    done
    shopt -u nullglob
    printf '%s\0' "${imgs[@]}"
  else
    printf '%s\0' "${path}"
  fi
}

BASE_URL="$(resolve_base_url)"
API_URL="${BASE_URL%/}/analyze"

if [[ "${RESTART_API}" == "true" ]]; then
  restart_api_8003 || true
  for _ in $(seq 1 20); do
    if curl -fsS "${BASE_URL%/}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
fi

process_one() {
  local image_path="$1"
  if [[ ! -f "${image_path}" ]]; then
    python3 - "${image_path}" "${TMP_JSON}" <<'PY'
import json
import os
import sys

image_path, out_path = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out_path), exist_ok=True)

j = {
    "errorCode": "IMAGE_NOT_FOUND",
    "message": f"画像パス \"{image_path}\" が見つかりません",
    "hint": "指定したパスが存在するか確認してください。",
    "debugImageBase64": "",
    "segments": [],
    "meta": {},
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(j, f, ensure_ascii=False)
PY
    local http_code="000"
    python3 - "${image_path}" "${TMP_JSON}" "${RUN_DIR}" "${OUT_DIR_BASE}" "${http_code}" <<'PY'
import base64
import binascii
import json
import os
import sys

image_path, in_path, run_dir, out_dir_base, http_code = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
base = os.path.splitext(os.path.basename(image_path))[0]

try:
    with open(in_path, "r", encoding="utf-8") as f:
        j = json.load(f)
except Exception:
    j = {"errorCode": f"HTTP_{http_code}", "message": "non-JSON response", "debugImageBase64": "", "segments": [], "meta": {}}

meta = j.get("meta")
if not isinstance(meta, dict):
    meta = {}

method = meta.get("segmentsMethod")
if not method and isinstance(meta.get("diagnostics"), dict):
    method = meta.get("diagnostics", {}).get("segmentsMethodChosen")
method = str(method or "unknown")

segments = j.get("segments")
seg_cnt = len(segments) if isinstance(segments, list) else 0

stem = f"{base}__{method}__seg{seg_cnt}"
out_json_path = os.path.join(run_dir, stem + ".json")
out_png_path = os.path.join(run_dir, stem + ".png")
summary_path = os.path.join(run_dir, stem + "__summary.txt")

with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(j, f, ensure_ascii=False, indent=2)

b64 = j.get("debugImageBase64") or ""
wrote_png = False
if b64:
    try:
        data = base64.b64decode(b64)
        with open(out_png_path, "wb") as f:
            f.write(data)
        wrote_png = True
    except binascii.Error:
        wrote_png = False

edges_b64 = ""
try:
    circle = meta.get("circle") if isinstance(meta, dict) else None
    if isinstance(circle, dict):
        cdiag = circle.get("diagnostics") if isinstance(circle.get("diagnostics"), dict) else {}
        edges_b64 = cdiag.get("edgesPngBase64") or ""
except Exception:
    edges_b64 = ""

latest_edges = os.path.join(out_dir_base, "latest_edges.png")
if edges_b64:
    try:
        data = base64.b64decode(edges_b64)
        with open(latest_edges, "wb") as f:
            f.write(data)
    except binascii.Error:
        pass

run_edges = os.path.join(run_dir, stem + "__edges.png")
if edges_b64:
    try:
        data = base64.b64decode(edges_b64)
        with open(run_edges, "wb") as f:
            f.write(data)
    except binascii.Error:
        pass

circle = meta.get("circle") if isinstance(meta, dict) else None
circle_method = "-"
circle_sanity = "-"
circle_cx = "-"
circle_cy = "-"
circle_r = "-"
if isinstance(circle, dict):
    circle_method = str(circle.get("method") or "-")
    circle_sanity = str(circle.get("sanityPassed") if circle.get("sanityPassed") is not None else "-")
    circle_cx = str(circle.get("cx") if circle.get("cx") is not None else "-")
    circle_cy = str(circle.get("cy") if circle.get("cy") is not None else "-")
    circle_r = str(circle.get("r") if circle.get("r") is not None else "-")

needle_time = str(meta.get("needleTimeHHMM") or "-")

error_code = str(j.get("errorCode") or "-")
message = str(j.get("message") or "-")

type_counts = {}
if isinstance(segments, list):
    for s in segments:
        if not isinstance(s, dict):
            continue
        t = str(s.get("type") or "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1
types_str = ",".join([f"{k}:{v}" for k, v in sorted(type_counts.items())]) if type_counts else "-"

summary = (
    f"image={image_path}\n"
    f"circle=({circle_method}) cx={circle_cx} cy={circle_cy} r={circle_r} sanity={circle_sanity}\n"
    f"needleTime={needle_time}\n"
    f"segmentsMethod={method} segCount={seg_cnt} types={types_str}\n"
    f"errorCode={error_code}\n"
    f"message={message}\n"
)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)

latest_json = os.path.join(out_dir_base, "latest.json")
latest_png = os.path.join(out_dir_base, "latest.png")
latest_summary = os.path.join(out_dir_base, "latest__summary.txt")
with open(latest_json, "w", encoding="utf-8") as f:
    json.dump(j, f, ensure_ascii=False, indent=2)
with open(latest_summary, "w", encoding="utf-8") as f:
    f.write(summary)
if wrote_png:
    with open(out_png_path, "rb") as src, open(latest_png, "wb") as dst:
        dst.write(src.read())

one_line = (
    f"{base} circle=({circle_method}) cx={circle_cx} cy={circle_cy} r={circle_r} sanity={circle_sanity} "
    f"needleTime={needle_time} segmentsMethod={method} segCount={seg_cnt} types={types_str} errorCode={error_code}"
)
print(one_line)

if wrote_png:
    print("__OPEN_IMAGE__=" + latest_png)
    print("__OPEN_DIR__=" + run_dir)
else:
    print("__OPEN_IMAGE__=" + image_path)
    print("__OPEN_DIR__=" + run_dir)
PY
    return 0
  fi

  local http_code
  http_code=$(curl -sS -o "${TMP_JSON}" -w "%{http_code}" -X POST "${API_URL}" \
    -F "file=@${image_path}" \
    -F "chartType=${CHART_TYPE}" \
    -F "midnightOffsetDeg=${MIDNIGHT_OFFSET_DEG}" || true)

  if [[ -z "${http_code}" ]]; then
    http_code="000"
  fi

  python3 - "${image_path}" "${TMP_JSON}" "${RUN_DIR}" "${OUT_DIR_BASE}" "${http_code}" <<'PY'
import base64
import binascii
import json
import os
import sys

image_path, in_path, run_dir, out_dir_base, http_code = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
base = os.path.splitext(os.path.basename(image_path))[0]

try:
    with open(in_path, "r", encoding="utf-8") as f:
        j = json.load(f)
except Exception:
    try:
        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            body = f.read(500)
    except Exception:
        body = ""
    j = {
        "errorCode": f"HTTP_{http_code}",
        "message": "non-JSON response" + (f" (HTTP {http_code})" if http_code else ""),
        "hint": body,
        "debugImageBase64": "",
        "segments": [],
        "meta": {},
    }

meta = j.get("meta")
if not isinstance(meta, dict):
    meta = {}

method = meta.get("segmentsMethod")
if not method and isinstance(meta.get("diagnostics"), dict):
    method = meta.get("diagnostics", {}).get("segmentsMethodChosen")
method = str(method or "unknown")

segments = j.get("segments")
seg_cnt = len(segments) if isinstance(segments, list) else 0

stem = f"{base}__{method}__seg{seg_cnt}"
out_json_path = os.path.join(run_dir, stem + ".json")
out_png_path = os.path.join(run_dir, stem + ".png")
summary_path = os.path.join(run_dir, stem + "__summary.txt")

with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(j, f, ensure_ascii=False, indent=2)

b64 = j.get("debugImageBase64") or ""
wrote_png = False
if b64:
    try:
        data = base64.b64decode(b64)
        with open(out_png_path, "wb") as f:
            f.write(data)
        wrote_png = True
    except binascii.Error:
        wrote_png = False

edges_b64 = ""
try:
    circle = meta.get("circle") if isinstance(meta, dict) else None
    if isinstance(circle, dict):
        cdiag = circle.get("diagnostics") if isinstance(circle.get("diagnostics"), dict) else {}
        edges_b64 = cdiag.get("edgesPngBase64") or ""
except Exception:
    edges_b64 = ""

latest_edges = os.path.join(out_dir_base, "latest_edges.png")
if edges_b64:
    try:
        data = base64.b64decode(edges_b64)
        with open(latest_edges, "wb") as f:
            f.write(data)
    except binascii.Error:
        pass

run_edges = os.path.join(run_dir, stem + "__edges.png")
if edges_b64:
    try:
        data = base64.b64decode(edges_b64)
        with open(run_edges, "wb") as f:
            f.write(data)
    except binascii.Error:
        pass

circle = meta.get("circle") if isinstance(meta, dict) else None
circle_method = "-"
circle_sanity = "-"
circle_cx = "-"
circle_cy = "-"
circle_r = "-"
if isinstance(circle, dict):
    circle_method = str(circle.get("method") or "-")
    circle_sanity = str(circle.get("sanityPassed") if circle.get("sanityPassed") is not None else "-")
    circle_cx = str(circle.get("cx") if circle.get("cx") is not None else "-")
    circle_cy = str(circle.get("cy") if circle.get("cy") is not None else "-")
    circle_r = str(circle.get("r") if circle.get("r") is not None else "-")

needle_time = str(meta.get("needleTimeHHMM") or "-")

error_code = str(j.get("errorCode") or "-")
message = str(j.get("message") or "-")

type_counts = {}
if isinstance(segments, list):
    for s in segments:
        if not isinstance(s, dict):
            continue
        t = str(s.get("type") or "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1
types_str = ",".join([f"{k}:{v}" for k, v in sorted(type_counts.items())]) if type_counts else "-"

summary = (
    f"image={image_path}\n"
    f"circle=({circle_method}) cx={circle_cx} cy={circle_cy} r={circle_r} sanity={circle_sanity}\n"
    f"needleTime={needle_time}\n"
    f"segmentsMethod={method} segCount={seg_cnt} types={types_str}\n"
    f"errorCode={error_code}\n"
    f"message={message}\n"
)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)

latest_json = os.path.join(out_dir_base, "latest.json")
latest_png = os.path.join(out_dir_base, "latest.png")
latest_summary = os.path.join(out_dir_base, "latest__summary.txt")
with open(latest_json, "w", encoding="utf-8") as f:
    json.dump(j, f, ensure_ascii=False, indent=2)
with open(latest_summary, "w", encoding="utf-8") as f:
    f.write(summary)
if wrote_png:
    with open(out_png_path, "rb") as src, open(latest_png, "wb") as dst:
        dst.write(src.read())

one_line = (
    f"{base} circle=({circle_method}) cx={circle_cx} cy={circle_cy} r={circle_r} sanity={circle_sanity} "
    f"needleTime={needle_time} segmentsMethod={method} segCount={seg_cnt} types={types_str} errorCode={error_code}"
)
print(one_line)

# Print tokens that the shell can parse for opening outputs.
if wrote_png:
    print("__OPEN_IMAGE__=" + latest_png)
    print("__OPEN_DIR__=" + run_dir)
else:
    # Fallback: open original input image, and also open the output directory in Finder.
    print("__OPEN_IMAGE__=" + image_path)
    print("__OPEN_DIR__=" + run_dir)
PY
}

open_path_if_any() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    return 0
  fi
  if command -v open >/dev/null 2>&1; then
    open "${path}" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${path}" >/dev/null 2>&1 || true
  fi
}

last_png=""
while IFS= read -r -d '' img; do
  out_lines="$(process_one "${img}" || true)"
  printf '%s\n' "${out_lines}" | sed -e '/^__OPEN_IMAGE__=/d' -e '/^__OPEN_DIR__=/d' || true
  open_img_line="$(printf '%s\n' "${out_lines}" | grep '^__OPEN_IMAGE__=' || true)"
  open_dir_line="$(printf '%s\n' "${out_lines}" | grep '^__OPEN_DIR__=' || true)"
  open_img_path="${open_img_line#__OPEN_IMAGE__=}"
  open_dir_path="${open_dir_line#__OPEN_DIR__=}"
  open_path_if_any "${open_img_path}"
  open_path_if_any "${open_dir_path}"
  echo ""
done < <(collect_images "${INPUT_PATH}")

if [[ "${OPEN_DIR}" == "true" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "${RUN_DIR}" >/dev/null 2>&1 || true
  fi
fi

echo "このフォルダを確認して: ${RUN_DIR}"
