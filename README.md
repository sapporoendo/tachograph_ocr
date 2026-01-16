# tachograph_ocr

## Demo (GitHub Pages)

URL:
`https://<your-github-username>.github.io/<repo_name>/`

This demo is built with `DEMO_MODE=true` and does not require the FastAPI backend.

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

Backend (in `backend/`):

```sh
python -m uvicorn app.main:app --reload --port 8000
```

Flutter:

```sh
flutter run -d chrome --dart-define=ANALYZER_API_BASE_URL=http://127.0.0.1:8000
```

## Build for GitHub Pages

```sh
flutter build web --release --dart-define=DEMO_MODE=true --base-href "/<repo_name>/"
```

## GitHub Actions deploy

- Workflow: `.github/workflows/deploy_demo.yml`
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
