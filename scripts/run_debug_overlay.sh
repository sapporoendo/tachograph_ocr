#!/usr/bin/env bash
set -euo pipefail

IMAGE_PATH="${1:-./sample.jpg}"
BASE_URL="${2:-}"
OUT_ROOT="${3:-}"
DEFAULT_BASE_URL="http://127.0.0.1:8000"

if [[ -z "${BASE_URL}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  DOTENV_PATH="${REPO_ROOT}/.env"

  if [[ -f "${DOTENV_PATH}" ]]; then
    env_base_url=$(
      python3 - "${DOTENV_PATH}" <<'PY'
import os
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
      BASE_URL="${env_base_url}"
    fi
  fi

  if [[ -z "${BASE_URL}" && -n "${API_BASE_URL:-}" ]]; then
    BASE_URL="${API_BASE_URL}"
  fi

  if [[ -z "${BASE_URL}" && -n "${ANALYZER_API_URL:-}" ]]; then
    BASE_URL="${ANALYZER_API_URL%/analyze}"
  fi

  if [[ -z "${BASE_URL}" ]]; then
    BASE_URL="${DEFAULT_BASE_URL}"
  fi
fi

API_URL="${BASE_URL%/}/analyze"

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="${HOME}/Downloads/tacho_debug"
fi

TS="$(date +"%Y-%m-%d_%H%M")"
OUT_DIR="${OUT_ROOT%/}/${TS}"
mkdir -p "${OUT_DIR}"
TMP_JSON="${OUT_DIR}/_response_tmp.json"

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "ERROR: image not found: $IMAGE_PATH" >&2
  echo "Usage: bash scripts/run_debug_overlay.sh /path/to/image.jpg [base_url] [out_root]" >&2
  echo "Example: bash scripts/run_debug_overlay.sh ./tachograph.jpg http://127.0.0.1:9000 ~/Downloads/tacho_debug" >&2
  exit 1
fi

http_code=$(curl -sS -o "$TMP_JSON" -w "%{http_code}" -X POST "$API_URL" \
  -F "file=@${IMAGE_PATH}" \
  -F "chartType=24h" \
  -F "midnightOffsetDeg=0")

if [[ "$http_code" != "200" ]]; then
  echo "ERROR: API returned HTTP $http_code" >&2
  echo "--- response body ($TMP_JSON) ---" >&2
  cat "$TMP_JSON" >&2 || true
  exit 1
fi

python3 - "$IMAGE_PATH" "$TMP_JSON" "$OUT_DIR" <<'PY'
import base64
import binascii
import json
import os
import sys

image_path, in_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

base = os.path.splitext(os.path.basename(image_path))[0]

with open(in_path, "r", encoding="utf-8") as f:
    j = json.load(f)

b64 = j.get("debugImageBase64") or ""
if not b64:
    raise SystemExit("ERROR: debugImageBase64 is empty in out.json")

method = "unknown"
meta = j.get("meta")
if isinstance(meta, dict) and meta.get("segmentsMethod"):
    method = str(meta.get("segmentsMethod"))

segments = j.get("segments")
seg_cnt = len(segments) if isinstance(segments, list) else 0

stem = f"{base}__{method}__seg{seg_cnt}"
out_json_path = os.path.join(out_dir, stem + ".json")
out_png_path = os.path.join(out_dir, stem + ".png")
summary_path = os.path.join(out_dir, stem + "__summary.txt")

try:
    data = base64.b64decode(b64)
except binascii.Error as e:
    raise SystemExit(f"ERROR: base64 decode failed: {e}")

with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(j, f, ensure_ascii=False, indent=2)

with open(out_png_path, "wb") as f:
    f.write(data)

circle = None
circle_method = "-"
circle_sanity = "-"
circle_cx = "-"
circle_cy = "-"
circle_r = "-"
if isinstance(meta, dict):
    circle = meta.get("circle")
if isinstance(circle, dict):
    circle_method = str(circle.get("method") or "-")
    circle_sanity = str(circle.get("sanityPassed") if circle.get("sanityPassed") is not None else "-")
    circle_cx = str(circle.get("cx") if circle.get("cx") is not None else "-")
    circle_cy = str(circle.get("cy") if circle.get("cy") is not None else "-")
    circle_r = str(circle.get("r") if circle.get("r") is not None else "-")

needle_time = "-"
if isinstance(meta, dict) and meta.get("needleTimeHHMM"):
    needle_time = str(meta.get("needleTimeHHMM"))

error_code = str(j.get("errorCode") or "-")

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
)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"wrote {out_json_path}")
print(f"wrote {out_png_path}")
print(f"wrote {summary_path}")
print(summary, end="")
PY

OUT_PNG="$(ls -1 "${OUT_DIR}"/*.png | tail -n 1)"
if command -v open >/dev/null 2>&1; then
  open "$OUT_PNG" >/dev/null 2>&1 || true
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$OUT_PNG" >/dev/null 2>&1 || true
fi
