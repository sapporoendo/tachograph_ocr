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

### Demo mode (no backend)

```sh
flutter pub get
flutter run -d chrome --dart-define=DEMO_MODE=true
```

### Normal mode (with backend)

Backend (in `api/`):

```sh
python3 -m pip install -r requirements.txt
python3 -m uvicorn api.main:app --reload --port 8000
```

Test with curl:

```sh
curl -s -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@/path/to/tachograph.jpg" \
  -F "chartType=24h" \
  -F "midnightOffsetDeg=0"
```

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

Flutter:

```sh
flutter run -d chrome --dart-define=ANALYZER_API_BASE_URL=http://127.0.0.1:8000
```

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

Important environment variables:

- `CORS_ALLOW_ORIGINS`
  - Comma-separated allowed origins
  - Example: `https://sapporoendo.github.io`
- `TACHO_DATA_DIR`
  - Data directory for SQLite DB + stored images/debug PNGs
