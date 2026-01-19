# tachograph_ocr

## Demo (GitHub Pages)

URL:
`https://sapporoendo.github.io/tachograph_ocr/`

This demo is built with `DEMO_MODE=true` and does not require the FastAPI backend.

## Full (GitHub Pages + External API)

To share the full-featured web app (uploads + history saved on server), build with:

- `DEMO_MODE=false`
- `SHOW_DEMO_TOGGLE=false`
- `ANALYZER_API_BASE_URL=https://<your-api-host>`

This repo includes a GitHub Actions workflow for this:

- `.github/workflows/deploy_pages.yml` (deploys to branch `gh-pages`)

### GitHub Variables

Set the following Repository Variable:

- `ANALYZER_API_BASE_URL`
  - Example: `https://tachograph-ocr-api.onrender.com`

### How to use

1. Open the demo URL
2. Select an image
3. Click `解析開始`
4. You will be redirected to the Result page with a fixed dummy result

## DEMO_MODE

- `DEMO_MODE=true`
  - The app never calls the API
  - Always returns a fixed dummy `AnalyzeResult`
- `DEMO_MODE=false` (default)
  - The app calls `POST /analyze` at `ANALYZER_API_BASE_URL`

## Local run

This section is the recommended way to share this repo with other developers.

Goal:

- Run FastAPI backend on your machine
- Run Flutter (web) frontend and connect it to your local backend

### Demo mode (no backend)

```sh
flutter pub get
flutter run -d chrome --dart-define=DEMO_MODE=true
```

### Normal mode (with backend)

#### 1) Backend (FastAPI)

Requirements:

- Python 3

Install dependencies and start the server (in repo root):

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Check health:

```sh
curl -s http://127.0.0.1:8000/health
```

Test with curl:

```sh
curl -s -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@/path/to/tachograph.jpg" \
  -F "chartType=24h" \
  -F "midnightOffsetDeg=0"
```

Notes:

- By default, CORS is permissive for local development.
- If you want to restrict CORS, set `CORS_ALLOW_ORIGINS` (comma-separated).

Generate debug overlay (out.json + debug_overlay.png):

```sh
bash scripts/run_debug_overlay.sh ./path/to/tachograph.jpg
bash scripts/run_debug_overlay.sh ./path/to/tachograph.jpg http://127.0.0.1:9000
bash scripts/run_debug_overlay.sh ./path/to/tachograph.jpg http://127.0.0.1:9000 ~/Downloads/tacho_debug
```

If you have `.env` in the repo root, `API_BASE_URL` will be used as the default base URL.

By default, outputs are written under `~/Downloads/tacho_debug/<YYYY-MM-DD_HHMM>/`.

Or:

```sh
make overlay IMAGE=./path/to/tachograph.jpg
```

#### 2) Frontend (Flutter)

In another terminal:

```sh
flutter pub get
flutter run -d chrome --dart-define=ANALYZER_API_BASE_URL=http://127.0.0.1:8000
```

Tips:

- If you want to hide the DEMO toggle in the UI:

```sh
flutter run -d chrome \
  --dart-define=SHOW_DEMO_TOGGLE=false \
  --dart-define=ANALYZER_API_BASE_URL=http://127.0.0.1:8000
```

- If you see a CORS error in Chrome, make sure the backend is running and you are using `http://127.0.0.1:8000` (no trailing slash).

## Build for GitHub Pages

```sh
flutter build web --release --dart-define=DEMO_MODE=true --base-href "/tachograph_ocr/" --no-tree-shake-icons
```

## GitHub Actions deploy

- Demo workflow: `.github/workflows/deploy_demo.yml`
  - Deploy target: `gh-pages-demo` branch (root)
- Full workflow: `.github/workflows/deploy_pages.yml`
  - Deploy target: `gh-pages` branch (root)

After the first successful workflow run:

1. Go to `Settings` -> `Pages`
2. Set `Source` to `Deploy from a branch`
3. Select branch `gh-pages` and folder `/ (root)`
4. Save

## Common pitfalls

- **base-href**
  - GitHub Pages requires `--base-href "/<repo_name>/"`
  - If it is wrong, you will see blank page or 404 for assets
- **404 on direct links**
  - GitHub Pages returns 404 for unknown paths
  - This workflow copies `index.html` to `404.html` to mitigate it
- **Cache / Service Worker**
  - A previous build can be cached
  - Try hard refresh, or open in incognito

## Deploy backend (Render)

This repo includes `render.yaml` for deploying the FastAPI backend.

This is optional for local development.

Important environment variables:

- `CORS_ALLOW_ORIGINS`
  - Comma-separated allowed origins
  - Example: `https://sapporoendo.github.io`
- `TACHO_DATA_DIR`
  - Data directory for SQLite DB + stored images/debug PNGs
