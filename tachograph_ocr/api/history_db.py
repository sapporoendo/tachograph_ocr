# Copyright (c) 2026 Kumiko Naito. All rights reserved.
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    return os.getenv("TACHO_DATA_DIR", os.path.join(repo_root, "data"))


def ensure_dirs(data_dir: str) -> Dict[str, str]:
    images_dir = os.path.join(data_dir, "images")
    debug_dir = os.path.join(data_dir, "debug")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    return {"data_dir": data_dir, "images_dir": images_dir, "debug_dir": debug_dir}


def _db_path(data_dir: str) -> str:
    return os.path.join(data_dir, "tachograph.sqlite3")


def connect(data_dir: Optional[str] = None) -> sqlite3.Connection:
    d = data_dir or _default_data_dir()
    ensure_dirs(d)
    conn = sqlite3.connect(_db_path(d))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
          id TEXT PRIMARY KEY,
          created_at TEXT NOT NULL,
          driver_name TEXT,
          vehicle_no TEXT,
          distance_km REAL,
          chart_type TEXT,
          midnight_offset_deg REAL,
          checked INTEGER NOT NULL DEFAULT 0,
          note TEXT,
          image_path TEXT NOT NULL,
          debug_image_path TEXT,
          analysis_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_records_created_at ON records(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_records_checked ON records(checked)")
    conn.commit()


def save_image_bytes(*, data_dir: str, filename_hint: str, image_bytes: bytes) -> Tuple[str, str]:
    dirs = ensure_dirs(data_dir)
    ext = os.path.splitext(filename_hint)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    rid = uuid.uuid4().hex
    rel = os.path.join("images", f"{rid}{ext}")
    abs_path = os.path.join(dirs["data_dir"], rel)
    with open(abs_path, "wb") as f:
        f.write(image_bytes)
    return rid, abs_path


def save_debug_png_bytes(*, data_dir: str, record_id: str, png_bytes: bytes) -> str:
    dirs = ensure_dirs(data_dir)
    rel = os.path.join("debug", f"{record_id}.png")
    abs_path = os.path.join(dirs["data_dir"], rel)
    with open(abs_path, "wb") as f:
        f.write(png_bytes)
    return abs_path


def insert_record(
    *,
    conn: sqlite3.Connection,
    record_id: str,
    driver_name: Optional[str],
    vehicle_no: Optional[str],
    distance_km: Optional[float],
    chart_type: Optional[str],
    midnight_offset_deg: Optional[float],
    image_path: str,
    debug_image_path: Optional[str],
    analysis_json: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO records(
          id, created_at, driver_name, vehicle_no, distance_km,
          chart_type, midnight_offset_deg, checked, note,
          image_path, debug_image_path, analysis_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?)
        """,
        (
            record_id,
            _now_iso(),
            driver_name,
            vehicle_no,
            distance_km,
            chart_type,
            midnight_offset_deg,
            image_path,
            debug_image_path,
            json.dumps(analysis_json, ensure_ascii=False),
        ),
    )
    conn.commit()


def list_records(
    *,
    conn: sqlite3.Connection,
    limit: int = 100,
    offset: int = 0,
    checked: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    where = ""
    args: List[Any] = []
    if checked is not None:
        where = "WHERE checked = ?"
        args.append(1 if checked else 0)

    rows = conn.execute(
        f"SELECT id, created_at, driver_name, vehicle_no, distance_km, chart_type, midnight_offset_deg, checked FROM records {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (*args, limit, offset),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "createdAt": r["created_at"],
                "driverName": r["driver_name"],
                "vehicleNo": r["vehicle_no"],
                "distanceKm": r["distance_km"],
                "chartType": r["chart_type"],
                "midnightOffsetDeg": r["midnight_offset_deg"],
                "checked": bool(r["checked"]),
            }
        )
    return out


def get_record(*, conn: sqlite3.Connection, record_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        return None

    analysis = json.loads(row["analysis_json"]) if row["analysis_json"] else {}
    return {
        "id": row["id"],
        "createdAt": row["created_at"],
        "driverName": row["driver_name"],
        "vehicleNo": row["vehicle_no"],
        "distanceKm": row["distance_km"],
        "chartType": row["chart_type"],
        "midnightOffsetDeg": row["midnight_offset_deg"],
        "checked": bool(row["checked"]),
        "note": row["note"],
        "imagePath": row["image_path"],
        "debugImagePath": row["debug_image_path"],
        "analysis": analysis,
    }


def set_checked(*, conn: sqlite3.Connection, record_id: str, checked: bool) -> bool:
    cur = conn.execute("UPDATE records SET checked = ? WHERE id = ?", (1 if checked else 0, record_id))
    conn.commit()
    return cur.rowcount > 0
