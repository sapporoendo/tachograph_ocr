from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DiscDetection:
    center_x: float
    center_y: float
    radius: float
    rotation_deg: Optional[float] = None


def detect_disc(image_bytes: bytes) -> Optional[DiscDetection]:
    _ = image_bytes
    return None
