import uvicorn
import sys
import os

# カレントディレクトリをパスに追加
sys.path.append(os.getcwd())

if __name__ == "__main__":
    # api/index.py が存在するか確認
    if os.path.exists("api/index.py"):
        uvicorn.run("api.index:app", host="127.0.0.1", port=8001, reload=True)
    elif os.path.exists("api/main.py"):
        uvicorn.run("api.main:app", host="127.0.0.1", port=8001, reload=True)
    else:
        print("エラー: api/index.py も api/main.py も見つかりません。")
        print("現在のフォルダの中身:", os.listdir("api") if os.path.exists("api") else "apiフォルダがありません")
