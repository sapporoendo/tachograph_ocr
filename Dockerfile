# 1. Pythonの公式イメージをベースにする
FROM python:3.9-slim

# 2. Tesseract OCRと日本語データをインストールする命令
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-jpn \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. 作業ディレクトリを作成
WORKDIR /app

# 4. 必要なファイルをコピー
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 5. Tesseractのパスを環境変数で指定（これが重要！）
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# 6. アプリを起動（ポート8000でFastAPIを起動）
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
