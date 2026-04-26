"""Domain types and project-file schema for pcbRE."""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass
from typing import Literal

Side = Literal["top", "bottom"]

PROJECT_VERSION = 3
PROJECT_EXT = ".pcbre"


@dataclass
class Point:
    """An alignment landmark: one pair forms a top↔bottom correspondence."""
    x: int
    y: int
    r: int


@dataclass
class Pad:
    """A user-named hole/pad in TOP-image coordinates.

    `side` records which layer was more visible when the user dropped it,
    purely as documentation — placement is in top space regardless.
    """
    x: int
    y: int
    r: int = 12
    name: str = ""
    description: str = ""
    color: str = "#ff3b30"
    opacity: float = 0.3
    side: Side = "top"


@dataclass
class Region:
    """Axis-aligned rectangle in TOP-image coordinates. Same metadata as Pad."""
    x: int  # center
    y: int  # center
    w: int = 80
    h: int = 50
    name: str = ""
    description: str = ""
    color: str = "#0a84ff"
    opacity: float = 0.3
    side: Side = "top"


def random_pad_color() -> str:
    """Saturated random color, evenly distributed over hue."""
    r, g, b = colorsys.hsv_to_rgb(random.random(), 0.85, 0.95)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


random_region_color = random_pad_color


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def normalize_side(value: str) -> Side:
    """Accept legacy 'bot' as well as canonical 'bottom' / 'top'."""
    return "bottom" if value in ("bot", "bottom") else "top"


def normalize_project_data(data: dict) -> dict:
    """Translate a v1 or v2 project document into a v2-shaped dict."""
    if int(data.get("version", 1)) >= 2:
        images = data.get("images") or {}
        view = data.get("view") or {}
        align = data.get("alignment_points") or {}
        out = {
            "images": {"top": images.get("top"), "bottom": images.get("bottom")},
            "bottom_mirror": bool(data.get("bottom_mirror", True)),
            "single_image": bool(data.get("single_image", False)),
            "view": {
                "rotation": int(view.get("rotation", 0)) % 360,
                "flipped": bool(view.get("flipped", False)),
                "opacity": float(view.get("opacity", 0.5)),
                "mode": str(view.get("mode", "overlay")),
            },
            "alignment_points": {
                "top": list(align.get("top") or []),
                "bottom": list(align.get("bottom") or []),
            },
            "pads": list(data.get("pads") or []),
            "regions": list(data.get("regions") or []),
        }
    else:  # legacy v1: flat keys, "bot_*" prefixes
        out = {
            "images": {"top": data.get("top_image"), "bottom": data.get("bot_image")},
            "bottom_mirror": bool(data.get("bot_mirror", True)),
            "single_image": False,
            "view": {
                "rotation": int(data.get("view_rotation", 0)) % 360,
                "flipped": bool(data.get("view_flipped", False)),
                "opacity": float(data.get("view_opacity", 0.5)),
                "mode": str(data.get("view_mode", "overlay")),
            },
            "alignment_points": {
                "top": list(data.get("top_points") or []),
                "bottom": list(data.get("bot_points") or []),
            },
            "pads": list(data.get("pads") or []),
            "regions": [],
        }
    if out["view"]["rotation"] not in (0, 90, 180, 270):
        out["view"]["rotation"] = 0
    return out
