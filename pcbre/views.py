"""Canvas widgets: zoomable base, alignment Panel, pad-aware OverlayView."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

from .imageops import BILINEAR, ROTATE_90, ROTATE_180, ROTATE_270
from .model import Pad, Point, Region, Side, hex_to_rgb, random_pad_color

PANEL_W, PANEL_H = 640, 720
OVERLAY_W, OVERLAY_H = 1300, 760
WHEEL_FACTOR = 1.18
# Cross-platform fallbacks for the pan/grab cursors. Tk on Aqua supports
# openhand/closedhand natively; X11 has hand1/hand2; "fleur" works everywhere.
PAN_CURSORS_OPEN = ("openhand", "hand2", "fleur")
PAN_CURSORS_GRAB = ("closedhand", "hand1", "fleur")
DEFAULT_CANVAS_CURSOR = "crosshair"


def set_canvas_cursor(canvas: tk.Canvas, candidates) -> None:
    """Set the first cursor name from `candidates` that the platform accepts."""
    for name in candidates if isinstance(candidates, tuple) else (candidates,):
        try:
            canvas.config(cursor=name)
            return
        except tk.TclError:
            continue
MIN_ZOOM = 0.05
MAX_ZOOM = 80.0
RADIUS_MIN = 2
RADIUS_MAX = 120
REGION_MIN = 4   # min width/height (top-image px) for a region to be committed
REGION_MAX = 8000

POINT_COLORS = [
    "#ff3b30", "#34c759", "#0a84ff", "#ff9f0a", "#bf5af2",
    "#ff375f", "#64d2ff", "#ffd60a", "#30d158", "#ac8e68",
    "#5e5ce6", "#ff6482", "#80ed99", "#ffadad", "#9bf6ff",
]

# Tk event.state mask for any of mouse buttons 1/2/3 held.
_BUTTON_MASK = 0x700


# ---------------------------------------------------------------------------
# ImageView: zoom, pan, fit, viewport math.
# ---------------------------------------------------------------------------

class ImageView:
    def __init__(self, parent: tk.Widget, width: int, height: int,
                 cursor: str = "") -> None:
        self.canvas = tk.Canvas(parent, bg="#1c1c1e",
                                width=width, height=height,
                                highlightthickness=0, cursor=cursor)
        self.zoom = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self._photo: ImageTk.PhotoImage | None = None
        self._pan_anchor: tuple[int, int, float, float] | None = None
        self._fitted_once = False
        self._pending = False

        c = self.canvas
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Button-4>", self._on_wheel)
        c.bind("<Button-5>", self._on_wheel)
        c.bind("<Double-Button-1>", lambda e: self.fit())
        c.bind("<Configure>", self._on_configure)
        # Right/middle button drag is handled by _pan_start/_pan_drag in the
        # subclass bindings; without an explicit release binding the
        # pan_anchor leaks and trips up the next gesture's release path.
        for ev in ("<ButtonRelease-2>", "<ButtonRelease-3>"):
            c.bind(ev, self._pan_end)

    def _pan_end(self, e=None) -> None:
        self._pan_anchor = None

    # Subclass hooks.
    def _content_size(self) -> tuple[int, int] | None: return None
    def _draw_content(self, ix0: int, iy0: int, ix1: int, iy1: int) -> None: ...
    def _draw_empty(self, cw: int, ch: int) -> None: ...
    def _draw_overlays(self) -> None: ...

    def pack(self, **kw): self.canvas.pack(**kw)
    def pack_forget(self): self.canvas.pack_forget()

    def _cw(self) -> int:
        w = int(self.canvas.winfo_width())
        return w if w > 1 else int(self.canvas.cget("width"))

    def _ch(self) -> int:
        h = int(self.canvas.winfo_height())
        return h if h > 1 else int(self.canvas.cget("height"))

    def canvas_to_orig(self, cx: float, cy: float) -> tuple[float, float]:
        return cx / self.zoom + self.ox, cy / self.zoom + self.oy

    def fit(self) -> None:
        size = self._content_size()
        if size is None:
            self.zoom = 1.0; self.ox = self.oy = 0.0
            self._fitted_once = True
            self.schedule_draw()
            return
        w, h = size
        cw, ch = self._cw(), self._ch()
        self.zoom = max(MIN_ZOOM, min(cw / w, ch / h))
        self.ox = (w - cw / self.zoom) / 2.0
        self.oy = (h - ch / self.zoom) / 2.0
        self._fitted_once = True
        self.schedule_draw()

    def schedule_draw(self) -> None:
        if self._pending:
            return
        self._pending = True
        self.canvas.after_idle(self._render)

    def _render(self) -> None:
        self._pending = False
        self.draw()

    def draw(self) -> None:
        c = self.canvas
        c.delete("all")
        cw, ch = self._cw(), self._ch()
        size = self._content_size()
        if size is None:
            self._draw_empty(cw, ch)
            return
        w, h = size
        ix0 = max(0, int(np.floor(self.ox)))
        iy0 = max(0, int(np.floor(self.oy)))
        ix1 = min(w, int(np.ceil(self.ox + cw / self.zoom)))
        iy1 = min(h, int(np.ceil(self.oy + ch / self.zoom)))
        if ix1 > ix0 and iy1 > iy0:
            self._draw_content(ix0, iy0, ix1, iy1)
        self._draw_overlays()

    def _on_wheel(self, e) -> None:
        if self._content_size() is None:
            return
        factor = _wheel_factor(e)
        if factor is None:
            return
        self._zoom_at(e.x, e.y, factor)

    def _zoom_at(self, cx: int, cy: int, factor: float) -> None:
        """Zoom by `factor` keeping the original-image pixel under (cx, cy) fixed."""
        sx, sy = self.canvas_to_orig(cx, cy)
        self.zoom = float(np.clip(self.zoom * factor, MIN_ZOOM, MAX_ZOOM))
        self.ox = sx - cx / self.zoom
        self.oy = sy - cy / self.zoom
        self.schedule_draw()

    def _pan_start(self, e) -> None:
        self._pan_anchor = (e.x, e.y, self.ox, self.oy)

    def _pan_drag(self, e) -> None:
        if self._pan_anchor is None or self._content_size() is None:
            return
        x0, y0, ox0, oy0 = self._pan_anchor
        self.ox = ox0 - (e.x - x0) / self.zoom
        self.oy = oy0 - (e.y - y0) / self.zoom
        self.schedule_draw()

    def _on_configure(self, e) -> None:
        if self._content_size() is None:
            return
        if not self._fitted_once and e.width > 1 and e.height > 1:
            self.fit()
        else:
            self.schedule_draw()


def _wheel_factor(e) -> float | None:
    """Map a Tk wheel event to a zoom factor, or None if it's not a tick."""
    delta = getattr(e, "delta", 0)
    num = getattr(e, "num", None)
    if delta > 0 or num == 4:
        return WHEEL_FACTOR
    if delta < 0 or num == 5:
        return 1.0 / WHEEL_FACTOR
    return None


def _wheel_sign(e) -> int:
    """+1 for up, -1 for down, 0 if it's not a tick."""
    delta = getattr(e, "delta", 0)
    num = getattr(e, "num", None)
    if delta > 0 or num == 4:
        return 1
    if delta < 0 or num == 5:
        return -1
    return 0


# ---------------------------------------------------------------------------
# LongPress: hold-to-place gesture. Dashed ring grows under the cursor and
# turns green when the threshold is crossed; release after that fires `ready`.
# ---------------------------------------------------------------------------

class LongPress:
    DURATION_MS = 250
    FRAMES = 5
    START_R = 6.0
    DEFAULT_TARGET_R = 24.0
    MIN_TARGET_R = 8.0

    def __init__(self, canvas: tk.Canvas) -> None:
        self.canvas = canvas
        self.active = False
        self.ready = False
        self.color = "#ffffff"
        self._cx = 0
        self._cy = 0
        self._step = 0
        self._tick_id: str | None = None
        self._target_r = self.DEFAULT_TARGET_R

    def start(self, cx: int, cy: int, color: str = "#ffffff",
              target_r: float | None = None) -> None:
        self.cancel()
        self.active = True
        self.ready = False
        self.color = color
        self._cx, self._cy = cx, cy
        self._step = 0
        self._target_r = max(self.MIN_TARGET_R,
                             float(target_r) if target_r is not None
                             else self.DEFAULT_TARGET_R)
        self._tick()

    def cancel(self) -> None:
        self.active = False
        self.ready = False
        try:
            self.canvas.delete("lp_ring")
        except tk.TclError:
            pass
        if self._tick_id is not None:
            try:
                self.canvas.after_cancel(self._tick_id)
            except tk.TclError:
                pass
            self._tick_id = None

    def _tick(self) -> None:
        if not self.active:
            return
        ready = self._step >= self.FRAMES
        progress = 1.0 if ready else self._step / self.FRAMES
        r = self.START_R + progress * (self._target_r - self.START_R)
        self.canvas.delete("lp_ring")
        self.canvas.create_oval(
            self._cx - r, self._cy - r, self._cx + r, self._cy + r,
            outline=self.color,
            width=3 if ready else 2,
            dash=() if ready else (3, 3),
            tag="lp_ring",
        )
        if ready:
            self.ready = True
            return
        self._step += 1
        self._tick_id = self.canvas.after(self.DURATION_MS // self.FRAMES, self._tick)


# ---------------------------------------------------------------------------
# Panel: alignment-side image with numbered points.
# ---------------------------------------------------------------------------

class Panel(ImageView):
    # Loose enough that trackpad press wobble doesn't kill a long-press hold.
    CLICK_THRESHOLD = 6

    def __init__(self, parent: tk.Widget, side: Side,
                 on_place: Callable[[Side, float, float], None],
                 on_grab:  Callable[[Side, int], None],
                 on_move:  Callable[[Side, int, float, float], None],
                 on_drop:  Callable[[Side, int], None],
                 on_resize: Callable[[Side, int, int], None] | None = None) -> None:
        super().__init__(parent, PANEL_W, PANEL_H, cursor="crosshair")
        self.side: Side = side
        self.on_place = on_place
        self.on_grab = on_grab
        self.on_move = on_move
        self.on_drop = on_drop
        self.on_resize = on_resize or (lambda s, i, r: None)

        self.image: Image.Image | None = None
        self.points: list[Point] = []
        self.selected_index: int | None = None

        self._press_orig: tuple[float, float] | None = None
        self._press_canvas: tuple[int, int] | None = None
        self._drag_idx: int | None = None
        self._drag_moved = False
        self._long_press = LongPress(self.canvas)
        # Set by App when the spacebar is held — primary-button drag pans the
        # canvas (Photoshop/Figma convention) instead of placing a point.
        self.space_held = False

        c = self.canvas
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_motion)
        c.bind("<ButtonRelease-1>", self._on_release)
        for ev in ("<ButtonPress-2>", "<ButtonPress-3>", "<Shift-ButtonPress-1>"):
            c.bind(ev, self._pan_start)
        for ev in ("<B2-Motion>", "<B3-Motion>", "<Shift-B1-Motion>"):
            c.bind(ev, self._pan_drag)

    def _content_size(self) -> tuple[int, int] | None:
        return (self.image.width, self.image.height) if self.image else None

    def set_image(self, img: Image.Image | None) -> None:
        self.image = img
        self._fitted_once = False
        self.canvas.update_idletasks()
        self.fit()

    def _hit(self, ox: float, oy: float) -> int | None:
        slop = 6.0 / max(self.zoom, 1e-6)
        for i in range(len(self.points) - 1, -1, -1):
            pt = self.points[i]
            if np.hypot(ox - pt.x, oy - pt.y) <= pt.r + slop:
                return i
        return None

    def _on_press(self, e) -> None:
        if self.image is None:
            return
        ox, oy = self.canvas_to_orig(e.x, e.y)
        hit = self._hit(ox, oy)
        if hit is not None:
            self._drag_idx = hit
            self._drag_moved = False
            self._press_orig = None
            self._press_canvas = None
            self._long_press.cancel()
            self.on_grab(self.side, hit)
            return
        self._drag_idx = None
        if self.space_held:
            # Space+drag = pan, no point placement.
            self._press_orig = None
            self._press_canvas = None
            self._pan_start(e)
            set_canvas_cursor(self.canvas, PAN_CURSORS_GRAB)
            return
        self._press_orig = (ox, oy)
        self._press_canvas = (e.x, e.y)
        color = POINT_COLORS[len(self.points) % len(POINT_COLORS)]
        self._long_press.start(e.x, e.y, color=color)

    def _on_motion(self, e) -> None:
        if self.image is None:
            return
        if self._drag_idx is not None:
            self._drag_moved = True
            ox, oy = self.canvas_to_orig(e.x, e.y)
            ox = float(np.clip(ox, 0, self.image.width - 1))
            oy = float(np.clip(oy, 0, self.image.height - 1))
            self.on_move(self.side, self._drag_idx, ox, oy)
            return
        if self._pan_anchor is not None:
            self._pan_drag(e)
            return
        if self._press_canvas is not None and self._long_press.active:
            cx, cy = self._press_canvas
            if abs(e.x - cx) > self.CLICK_THRESHOLD or abs(e.y - cy) > self.CLICK_THRESHOLD:
                self._long_press.cancel()
                self._press_orig = None
                self._press_canvas = None

    def _on_release(self, e) -> None:
        if self._drag_idx is not None:
            if self._drag_moved:
                self.on_drop(self.side, self._drag_idx)
            self._drag_idx = None
            return
        if self._pan_anchor is not None:
            self._pan_anchor = None
            # If space is still held the user is staged for another pan, so
            # show the open hand again; otherwise restore the default cursor.
            set_canvas_cursor(
                self.canvas,
                PAN_CURSORS_OPEN if self.space_held else DEFAULT_CANVAS_CURSOR)
            return
        if self._press_orig is not None and self._long_press.ready:
            ox, oy = self._press_orig
            self.on_place(self.side, ox, oy)
        self._long_press.cancel()
        self._press_orig = None
        self._press_canvas = None

    def _on_wheel(self, e) -> None:
        # Hold a point + scroll → resize that point instead of zooming.
        if self._drag_idx is not None and self._drag_idx < len(self.points):
            sign = _wheel_sign(e)
            if sign == 0:
                return
            pt = self.points[self._drag_idx]
            step = max(1, pt.r // 10)
            new_r = max(RADIUS_MIN, min(RADIUS_MAX, pt.r + sign * step))
            if new_r != pt.r:
                self.on_resize(self.side, self._drag_idx, new_r)
            return
        super()._on_wheel(e)

    def _draw_empty(self, cw: int, ch: int) -> None:
        self.canvas.create_text(cw // 2, ch // 2, fill="#888",
                                text=f"Load {self.side} image",
                                font=("TkDefaultFont", 12))

    def _draw_content(self, ix0: int, iy0: int, ix1: int, iy1: int) -> None:
        crop = self.image.crop((ix0, iy0, ix1, iy1))
        new_w = max(1, int(round((ix1 - ix0) * self.zoom)))
        new_h = max(1, int(round((iy1 - iy0) * self.zoom)))
        self._photo = ImageTk.PhotoImage(crop.resize((new_w, new_h), BILINEAR))
        cx0 = (ix0 - self.ox) * self.zoom
        cy0 = (iy0 - self.oy) * self.zoom
        self.canvas.create_image(cx0, cy0, anchor="nw", image=self._photo)

    def _draw_overlays(self) -> None:
        c = self.canvas
        for i, pt in enumerate(self.points):
            color = POINT_COLORS[i % len(POINT_COLORS)]
            X = (pt.x - self.ox) * self.zoom
            Y = (pt.y - self.oy) * self.zoom
            R = max(3.0, pt.r * self.zoom)
            if i == self.selected_index:
                c.create_oval(X - R - 4, Y - R - 4, X + R + 4, Y + R + 4,
                              outline="white", width=2)
                c.create_line(X - R - 6, Y, X + R + 6, Y, fill="white")
                c.create_line(X, Y - R - 6, X, Y + R + 6, fill="white")
            c.create_oval(X - R, Y - R, X + R, Y + R, outline=color, width=2)
            c.create_oval(X - 3, Y - 3, X + 3, Y + 3, fill=color, outline="white")
            label = f"{i + 1}  r={pt.r}px" if i == self.selected_index else str(i + 1)
            c.create_text(X + R + 8, Y, text=label, fill=color, anchor="w",
                          font=("TkDefaultFont", 10, "bold"))


# ---------------------------------------------------------------------------
# Tooltip drawn into the canvas (sharp corners on every platform).
# ---------------------------------------------------------------------------

class Tooltip:
    DELAY_MS = 400

    def __init__(self, canvas: tk.Canvas) -> None:
        self.canvas = canvas
        self.active = False
        self.text = ""
        self.x = 0
        self.y = 0
        self._after_id: str | None = None

    def schedule(self, x: int, y: int, text: str,
                 on_show: Callable[[], None] | None = None) -> None:
        self.cancel()
        if not text:
            return
        self._after_id = self.canvas.after(
            self.DELAY_MS, lambda: self._activate(x, y, text, on_show))

    def cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.canvas.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def hide(self, on_change: Callable[[], None] | None = None) -> None:
        self.cancel()
        was_active = self.active
        self.active = False
        if was_active and on_change is not None:
            on_change()

    def _activate(self, x: int, y: int, text: str,
                  on_show: Callable[[], None] | None) -> None:
        self._after_id = None
        self.active = True
        self.text = text
        self.x = x
        self.y = y
        if on_show is not None:
            on_show()

    def draw_into(self) -> None:
        if not self.active:
            return
        c = self.canvas
        text_id = c.create_text(
            self.x + 14, self.y + 18, text=self.text, anchor="nw",
            fill="#000000", justify="left", font=("TkDefaultFont", 10))
        bbox = c.bbox(text_id)
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        pad = 4
        bx0, by0, bx1, by1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        cw, ch = int(c.winfo_width()), int(c.winfo_height())
        dx, dy = max(0, bx1 - cw), max(0, by1 - ch)
        if dx or dy:
            c.move(text_id, -dx, -dy)
            bx0 -= dx; bx1 -= dx; by0 -= dy; by1 -= dy
        bg_id = c.create_rectangle(bx0, by0, bx1, by1,
                                   fill="#ffffe0", outline="#888888")
        c.tag_lower(bg_id, text_id)


# ---------------------------------------------------------------------------
# OverlayView: blended top+warped (or single source) with pad layer.
# ---------------------------------------------------------------------------

class OverlayView(ImageView):
    # Loose enough that trackpad press wobble doesn't kill a long-press hold.
    CLICK_THRESHOLD = 6
    HANDLE_R = 5  # screen-px radius of the resize dots on a selected region

    def __init__(self, parent: tk.Widget,
                 on_place_pad:        Callable[[float, float, Side, str], None] | None = None,
                 on_grab_pad:         Callable[[int], None] | None = None,
                 on_move_pad:         Callable[[int, float, float], None] | None = None,
                 on_drop_pad:         Callable[[int], None] | None = None,
                 on_resize_pad:       Callable[[int, int], None] | None = None,
                 on_pad_deselect:     Callable[[], None] | None = None,
                 on_double_click_pad: Callable[[int], None] | None = None,
                 on_pad_toggle:       Callable[[int], None] | None = None,
                 on_place_region:        Callable[[float, float, float, float, Side, str], None] | None = None,
                 on_grab_region:         Callable[[int], None] | None = None,
                 on_move_region:         Callable[[int, float, float], None] | None = None,
                 on_resize_region:       Callable[[int, float, float, float, float], None] | None = None,
                 on_region_deselect:     Callable[[], None] | None = None,
                 on_double_click_region: Callable[[int], None] | None = None,
                 single_source: str | None = None) -> None:
        super().__init__(parent, OVERLAY_W, OVERLAY_H, cursor="crosshair")
        self.top: Image.Image | None = None
        self.warped: Image.Image | None = None
        self._top_view: Image.Image | None = None
        self._warped_view: Image.Image | None = None
        self.alpha = 0.5
        self.rotation = 0
        self.flipped = False
        self.single_source = single_source  # None | "top" | "warped"
        # Set by App when the spacebar is held — primary-button drag pans the
        # canvas (Photoshop/Figma convention) instead of starting a region.
        self.space_held = False

        # Pads live in TOP-image coordinates so they survive view rotation/flip.
        self.pads: list[Pad] = []
        self.selected_pad: int | None = None
        # Multi-selection: indices of all selected pads. `selected_pad` mirrors
        # the only entry when len == 1, and is None otherwise (the editor flow
        # depends on a single primary).
        self.selected_pads: set[int] = set()
        self.on_place_pad = on_place_pad or (lambda x, y, s, c: None)
        self.on_grab_pad = on_grab_pad or (lambda i: None)
        self.on_move_pad = on_move_pad or (lambda i, x, y: None)
        self.on_drop_pad = on_drop_pad or (lambda i: None)
        self.on_resize_pad = on_resize_pad or (lambda i, r: None)
        self.on_pad_deselect = on_pad_deselect or (lambda: None)
        self.on_double_click_pad = on_double_click_pad or (lambda i: None)
        self.on_pad_toggle = on_pad_toggle or (lambda i: None)

        # Regions: same TOP-coord storage; rectangles stay axis-aligned through
        # the 90°/flip view transforms used here.
        self.regions: list[Region] = []
        self.selected_region: int | None = None
        # Top-image radius used by the long-press preview ring so the growing
        # ring lands at the same on-screen size as the pad about to be dropped.
        self.next_pad_radius: int = 12
        self.on_place_region = on_place_region or (lambda x, y, w, h, s, c: None)
        self.on_grab_region = on_grab_region or (lambda i: None)
        self.on_move_region = on_move_region or (lambda i, x, y: None)
        self.on_resize_region = on_resize_region or (lambda i, x, y, w, h: None)
        self.on_region_deselect = on_region_deselect or (lambda: None)
        self.on_double_click_region = on_double_click_region or (lambda i: None)

        # Cache for the cropped-and-resized base images. The resize from a
        # multi-megapixel source to canvas resolution is the dominant cost per
        # frame, and it doesn't depend on alpha — so we reuse it across opacity
        # drags. Keyed on view bounds + zoom + image identity.
        self._scaled_cache_key: tuple | None = None
        self._scaled_cache_top: Image.Image | None = None
        self._scaled_cache_warped: Image.Image | None = None

        self._press_view: tuple[float, float] | None = None
        self._press_canvas: tuple[int, int] | None = None
        self._drag_pad: int | None = None
        # Press point + original pad center, so the drag preserves the
        # cursor's grab offset (no "jump to center" on first motion).
        self._drag_pad_press_top: tuple[float, float] | None = None
        self._drag_pad_orig_xy: tuple[int, int] | None = None
        # Group drag: indices, press point in top coords, and each pad's
        # original (x, y) so we can apply a uniform delta on motion.
        self._drag_pads: list[int] | None = None
        self._drag_pads_press_top: tuple[float, float] | None = None
        self._drag_pads_orig_xy: list[tuple[int, int]] | None = None
        self._drag_region: int | None = None
        self._region_resize_idx: int | None = None
        self._region_resize_handle: str | None = None
        # View-coord rect at the start of a resize, so we can update by deltas.
        self._region_resize_start_rect: tuple[float, float, float, float] | None = None
        # In-progress region creation, both endpoints in view coords.
        self._region_create_start: tuple[float, float] | None = None
        self._region_create_end: tuple[float, float] | None = None
        self._pending_region_color = "#0a84ff"
        self._click_moved = False
        self._tooltip = Tooltip(self.canvas)
        self._hover_idx: int | None = None
        self._long_press = LongPress(self.canvas)
        self._pending_pad_color = "#ff3b30"

        c = self.canvas
        c.bind("<ButtonPress-1>",   self._on_press)
        c.bind("<B1-Motion>",       self._on_motion)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Double-Button-1>", self._on_double_click)
        c.bind("<Motion>",          self._on_hover)
        c.bind("<Leave>",           lambda e: self._on_leave())
        c.bind("<Alt-ButtonPress-1>", self._on_alt_press)
        for ev in ("<Alt-MouseWheel>", "<Alt-Button-4>", "<Alt-Button-5>"):
            c.bind(ev, self._on_alt_wheel)
        # Shift+click pad toggles in multi-selection; Shift+drag on empty
        # still pans (one handler dispatches both).
        c.bind("<Shift-ButtonPress-1>", self._on_shift_press)
        # Cmd/Ctrl+click pad: pad-only toggle (macOS / Windows convention).
        c.bind("<Command-ButtonPress-1>", self._on_toggle_press)
        c.bind("<Control-ButtonPress-1>", self._on_toggle_press)
        for ev in ("<ButtonPress-2>", "<ButtonPress-3>"):
            c.bind(ev, self._pan_start)
        for ev in ("<B2-Motion>", "<B3-Motion>", "<Shift-B1-Motion>"):
            c.bind(ev, self._pan_drag)

    def _on_double_click(self, e) -> None:
        if self._content_size() is None:
            self.fit()
            return
        vx, vy = self.canvas_to_orig(e.x, e.y)
        pad_hit = self._hit_pad(vx, vy)
        if pad_hit is not None:
            self._tooltip.hide(self.schedule_draw)
            self.on_double_click_pad(pad_hit)
            return
        region_hit = self._hit_region_body(vx, vy)
        if region_hit is not None:
            self._tooltip.hide(self.schedule_draw)
            self.on_double_click_region(region_hit)
            return
        self.fit()

    # -- view content --------------------------------------------------------

    def _content_size(self) -> tuple[int, int] | None:
        view = self._active_view()
        return view.size if view is not None else None

    def _active_view(self) -> Image.Image | None:
        if self.single_source == "top":
            return self._top_view
        if self.single_source == "warped":
            return self._warped_view
        if self._top_view is None or self._warped_view is None:
            return None
        return self._top_view

    def _detect_side(self) -> Side:
        if self.single_source == "top":
            return "top"
        if self.single_source == "warped":
            return "bottom"
        return "bottom" if self.alpha > 0.5 else "top"

    def _side_visibility(self, side: Side) -> float:
        """0..1 visibility factor for a pad/region on `side` in this view.

        Single-source views show only their own side. Single-image projects
        (where top and warped reference the same image) skip the fade entirely.
        Otherwise we ramp the layer's effective opacity 0.5 → 0.6 (hidden →
        shown) so the slider can keep one side, the other, or neither.
        """
        if self.single_source == "top":
            return 1.0 if side == "top" else 0.0
        if self.single_source == "warped":
            return 1.0 if side == "bottom" else 0.0
        if self.top is not None and self.top is self.warped:
            return 1.0
        layer_op = self.alpha if side == "bottom" else (1.0 - self.alpha)
        if layer_op <= 0.5:
            return 0.0
        if layer_op >= 0.6:
            return 1.0
        return (layer_op - 0.5) / 0.1

    @staticmethod
    def _fade_color(hex_color: str, factor: float) -> str:
        """Blend `hex_color` toward the canvas bg by `factor` (1 = original)."""
        if factor >= 0.999:
            return hex_color
        r, g, b = hex_to_rgb(hex_color)
        bg_r, bg_g, bg_b = 0x1c, 0x1c, 0x1e
        nr = int(round(bg_r + (r - bg_r) * factor))
        ng = int(round(bg_g + (g - bg_g) * factor))
        nb = int(round(bg_b + (b - bg_b) * factor))
        return f"#{nr:02x}{ng:02x}{nb:02x}"

    def set_pair(self, top: Image.Image | None,
                 warped: Image.Image | None) -> None:
        self.top = top
        self.warped = warped
        self._rebuild_views()
        self._invalidate_scaled_cache()
        self._fitted_once = False
        self.canvas.update_idletasks()
        self.fit()

    def set_alpha(self, a: float) -> None:
        self.alpha = float(a)
        # Single-source views don't blend, and side visibility is independent
        # of alpha there, so we don't need to redraw them on opacity changes.
        if self.single_source is None:
            self.schedule_draw()

    def _invalidate_scaled_cache(self) -> None:
        self._scaled_cache_key = None
        self._scaled_cache_top = None
        self._scaled_cache_warped = None

    def rotate_cw(self) -> None:
        self.rotation = (self.rotation + 90) % 360
        self._reapply_view_transform()

    def toggle_flip(self) -> None:
        self.flipped = not self.flipped
        self._reapply_view_transform()

    def _reapply_view_transform(self) -> None:
        if self.top is None or self.warped is None:
            return
        self._rebuild_views()
        self._invalidate_scaled_cache()
        self._fitted_once = False
        self.fit()

    def _rebuild_views(self) -> None:
        self._top_view = self._apply(self.top)
        self._warped_view = self._apply(self.warped)

    def _apply(self, img: Image.Image | None) -> Image.Image | None:
        if img is None:
            return None
        out = ImageOps.mirror(img) if self.flipped else img
        # Pillow's ROTATE_n is counter-clockwise; we expose clockwise.
        if self.rotation == 90:
            out = out.transpose(ROTATE_270)
        elif self.rotation == 180:
            out = out.transpose(ROTATE_180)
        elif self.rotation == 270:
            out = out.transpose(ROTATE_90)
        return out

    # -- coord transforms ----------------------------------------------------

    def top_to_view(self, x: float, y: float) -> tuple[float, float]:
        if self.top is None:
            return x, y
        W, H = self.top.size
        if self.flipped:
            x = (W - 1) - x
        if self.rotation == 90:
            return (H - 1) - y, x
        if self.rotation == 180:
            return (W - 1) - x, (H - 1) - y
        if self.rotation == 270:
            return y, (W - 1) - x
        return x, y

    def view_to_top(self, vx: float, vy: float) -> tuple[float, float]:
        if self.top is None:
            return vx, vy
        W, H = self.top.size
        if self.rotation == 90:
            x, y = vy, (H - 1) - vx
        elif self.rotation == 180:
            x, y = (W - 1) - vx, (H - 1) - vy
        elif self.rotation == 270:
            x, y = (W - 1) - vy, vx
        else:
            x, y = vx, vy
        if self.flipped:
            x = (W - 1) - x
        return x, y

    # -- region geometry -----------------------------------------------------

    def _region_view_rect(self, r: Region) -> tuple[float, float, float, float]:
        """Region's axis-aligned bounding box in *view* coords (vx0, vy0, vx1, vy1)."""
        half_w = r.w / 2.0
        half_h = r.h / 2.0
        corners = (
            (r.x - half_w, r.y - half_h), (r.x + half_w, r.y - half_h),
            (r.x + half_w, r.y + half_h), (r.x - half_w, r.y + half_h),
        )
        vxs = []
        vys = []
        for tx, ty in corners:
            vx, vy = self.top_to_view(tx, ty)
            vxs.append(vx)
            vys.append(vy)
        return min(vxs), min(vys), max(vxs), max(vys)

    def _view_rect_to_top_rect(self, vx0: float, vy0: float,
                                vx1: float, vy1: float) -> tuple[float, float, float, float]:
        """Inverse of `_region_view_rect`. Returns (cx, cy, w, h) in top coords."""
        corners = (
            self.view_to_top(vx0, vy0), self.view_to_top(vx1, vy0),
            self.view_to_top(vx1, vy1), self.view_to_top(vx0, vy1),
        )
        txs = [c[0] for c in corners]
        tys = [c[1] for c in corners]
        tx0, tx1 = min(txs), max(txs)
        ty0, ty1 = min(tys), max(tys)
        return (tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0, tx1 - tx0, ty1 - ty0

    def _hit_region_body(self, vx: float, vy: float) -> int | None:
        for i in range(len(self.regions) - 1, -1, -1):
            reg = self.regions[i]
            if self._side_visibility(reg.side) <= 0.0:
                continue
            rx0, ry0, rx1, ry1 = self._region_view_rect(reg)
            if rx0 <= vx <= rx1 and ry0 <= vy <= ry1:
                return i
        return None

    def _region_handle_at(self, vx: float, vy: float) -> str | None:
        idx = self.selected_region
        if idx is None or idx >= len(self.regions):
            return None
        rx0, ry0, rx1, ry1 = self._region_view_rect(self.regions[idx])
        cxv = (rx0 + rx1) / 2.0
        cyv = (ry0 + ry1) / 2.0
        slop = (self.HANDLE_R + 4) / max(self.zoom, 1e-6)
        handles = (
            ("nw", rx0, ry0), ("n", cxv, ry0), ("ne", rx1, ry0),
            ("w",  rx0, cyv),                  ("e",  rx1, cyv),
            ("sw", rx0, ry1), ("s", cxv, ry1), ("se", rx1, ry1),
        )
        for name, hx, hy in handles:
            if abs(vx - hx) <= slop and abs(vy - hy) <= slop:
                return name
        return None

    # -- input ---------------------------------------------------------------

    def _hit_pad(self, vx: float, vy: float) -> int | None:
        if not self.pads:
            return None
        slop = 4.0 / max(self.zoom, 1e-6)
        for i in range(len(self.pads) - 1, -1, -1):
            p = self.pads[i]
            if self._side_visibility(p.side) <= 0.0:
                continue
            pvx, pvy = self.top_to_view(p.x, p.y)
            if np.hypot(vx - pvx, vy - pvy) <= p.r + slop:
                return i
        return None

    def _on_press(self, e) -> None:
        if self._content_size() is None:
            return
        vx, vy = self.canvas_to_orig(e.x, e.y)
        self._click_moved = False
        self._tooltip.hide(self.schedule_draw)

        # Resize takes priority over body/pad hits when a region is already
        # selected — the dots are only there for the selected region anyway.
        if self.selected_region is not None:
            handle = self._region_handle_at(vx, vy)
            if handle is not None:
                self._region_resize_idx = self.selected_region
                self._region_resize_handle = handle
                self._region_resize_start_rect = self._region_view_rect(
                    self.regions[self.selected_region])
                self._press_view = (vx, vy)
                return

        pad_hit = self._hit_pad(vx, vy)
        if pad_hit is not None:
            # Pad already part of a multi-selection → drag the group, keep
            # the selection intact (don't fire on_grab_pad which would
            # collapse the set down to this single pad).
            if pad_hit in self.selected_pads and len(self.selected_pads) > 1:
                self._drag_pads = sorted(self.selected_pads)
                self._drag_pads_press_top = self.view_to_top(vx, vy)
                self._drag_pads_orig_xy = [
                    (self.pads[i].x, self.pads[i].y) for i in self._drag_pads]
                self._press_view = (vx, vy)
                return
            self._drag_pad = pad_hit
            self._drag_pad_press_top = self.view_to_top(vx, vy)
            self._drag_pad_orig_xy = (
                self.pads[pad_hit].x, self.pads[pad_hit].y)
            self._press_view = None
            if self.selected_region is not None:
                self.selected_region = None
                self.on_region_deselect()
            self.on_grab_pad(pad_hit)
            return

        region_hit = self._hit_region_body(vx, vy)
        if region_hit is not None:
            self._drag_region = region_hit
            self._press_view = (vx, vy)
            self._region_drag_press_top = self.view_to_top(vx, vy)
            self._region_drag_orig_xy = (
                self.regions[region_hit].x, self.regions[region_hit].y)
            if self.selected_pad is not None:
                self.selected_pad = None
                self.on_pad_deselect()
            self.on_grab_region(region_hit)
            return

        # Empty space: drop any prior selection (single or multi).
        if self.selected_pad is not None or self.selected_pads:
            self.selected_pad = None
            self.selected_pads = set()
            self.on_pad_deselect()
        if self.selected_region is not None:
            self.selected_region = None
            self.on_region_deselect()

        # Spacebar held → drag pans the canvas.
        if self.space_held:
            self._pan_start(e)
            set_canvas_cursor(self.canvas, PAN_CURSORS_GRAB)
            return

        # Default: arm long-press for pad placement and remember the press
        # position so a motion past CLICK_THRESHOLD can promote the gesture
        # into region creation.
        self._press_view = (vx, vy)
        self._press_canvas = (e.x, e.y)
        self._pending_pad_color = random_pad_color()
        self._pending_region_color = self._pending_pad_color
        target_r = self.next_pad_radius * self.zoom
        self._long_press.start(e.x, e.y, color=self._pending_pad_color,
                               target_r=target_r)

    def _on_shift_press(self, e) -> None:
        """Shift+click on a pad → toggle in multi-selection. On empty space
        it falls through to a pan, matching the existing Shift+drag pan."""
        if self._content_size() is None:
            return
        vx, vy = self.canvas_to_orig(e.x, e.y)
        pad_hit = self._hit_pad(vx, vy)
        if pad_hit is not None:
            self._tooltip.hide(self.schedule_draw)
            self.on_pad_toggle(pad_hit)
            return
        self._pan_start(e)
        set_canvas_cursor(self.canvas, PAN_CURSORS_GRAB)

    def _on_toggle_press(self, e) -> None:
        """Cmd/Ctrl+click on a pad → toggle in multi-selection."""
        if self._content_size() is None:
            return
        vx, vy = self.canvas_to_orig(e.x, e.y)
        pad_hit = self._hit_pad(vx, vy)
        if pad_hit is not None:
            self._tooltip.hide(self.schedule_draw)
            self.on_pad_toggle(pad_hit)

    def _on_alt_press(self, e) -> None:
        """Alt+click on a selected region's edge dot starts a resize."""
        if self._content_size() is None:
            return
        vx, vy = self.canvas_to_orig(e.x, e.y)
        handle = self._region_handle_at(vx, vy)
        if handle is None or self.selected_region is None:
            # Nothing to do for Alt+click elsewhere — fall through to normal press.
            self._on_press(e)
            return
        self._tooltip.hide(self.schedule_draw)
        self._click_moved = False
        self._region_resize_idx = self.selected_region
        self._region_resize_handle = handle
        self._region_resize_start_rect = self._region_view_rect(
            self.regions[self.selected_region])
        self._press_view = (vx, vy)

    def _on_motion(self, e) -> None:
        if self._content_size() is None:
            return
        if self._region_resize_handle is not None:
            self._update_region_resize(e)
            return
        if self._drag_region is not None:
            self._click_moved = True
            vx, vy = self.canvas_to_orig(e.x, e.y)
            tx, ty = self.view_to_top(vx, vy)
            dx = tx - self._region_drag_press_top[0]
            dy = ty - self._region_drag_press_top[1]
            new_x = self._region_drag_orig_xy[0] + dx
            new_y = self._region_drag_orig_xy[1] + dy
            self.on_move_region(self._drag_region, new_x, new_y)
            return
        if self._drag_pads is not None:
            self._click_moved = True
            vx, vy = self.canvas_to_orig(e.x, e.y)
            tx, ty = self.view_to_top(vx, vy)
            dx = tx - self._drag_pads_press_top[0]
            dy = ty - self._drag_pads_press_top[1]
            for idx, (ox, oy) in zip(self._drag_pads, self._drag_pads_orig_xy):
                self.on_move_pad(idx, ox + dx, oy + dy)
            return
        if self._drag_pad is not None:
            self._click_moved = True
            vx, vy = self.canvas_to_orig(e.x, e.y)
            tx, ty = self.view_to_top(vx, vy)
            dx = tx - self._drag_pad_press_top[0]
            dy = ty - self._drag_pad_press_top[1]
            self.on_move_pad(self._drag_pad,
                             self._drag_pad_orig_xy[0] + dx,
                             self._drag_pad_orig_xy[1] + dy)
            return
        if self._region_create_start is not None:
            self._click_moved = True
            self._region_create_end = self.canvas_to_orig(e.x, e.y)
            self.schedule_draw()
            return
        if self._pan_anchor is not None:
            self._click_moved = True
            self._pan_drag(e)
            return
        # Default empty press: motion past the click threshold cancels the
        # long-press and starts region creation from the original press point.
        if self._press_canvas is not None and self._press_view is not None:
            x0, y0 = self._press_canvas
            if abs(e.x - x0) <= self.CLICK_THRESHOLD and abs(e.y - y0) <= self.CLICK_THRESHOLD:
                return
            # Once the long-press has filled, the pad is locked in — late
            # wobble shouldn't downgrade an already-armed hold into a region.
            if self._long_press.ready:
                return
            self._click_moved = True
            self._long_press.cancel()
            self._region_create_start = self._press_view
            self._region_create_end = self.canvas_to_orig(e.x, e.y)
            self._press_canvas = None
            self.schedule_draw()

    def _on_release(self, e) -> None:
        # Region resize: notify app once at end so any clamping/bookkeeping fires.
        if self._region_resize_handle is not None:
            idx = self._region_resize_idx
            if idx is not None and idx < len(self.regions):
                r = self.regions[idx]
                self.on_resize_region(idx, r.x, r.y, r.w, r.h)
            self._region_resize_handle = None
            self._region_resize_idx = None
            self._region_resize_start_rect = None
            self._press_view = None
            self._press_canvas = None
            return

        # Region drag: moves were applied live; nothing extra to commit here.
        if self._drag_region is not None:
            self._drag_region = None
            self._press_view = None
            self._press_canvas = None
            return

        # Group pad drag: each pad already moved live via on_move_pad.
        if self._drag_pads is not None:
            if self._click_moved:
                for idx in self._drag_pads:
                    self.on_drop_pad(idx)
            self._drag_pads = None
            self._drag_pads_press_top = None
            self._drag_pads_orig_xy = None
            self._press_view = None
            self._press_canvas = None
            return

        if self._drag_pad is not None:
            if self._click_moved:
                self.on_drop_pad(self._drag_pad)
            self._drag_pad = None
            self._drag_pad_press_top = None
            self._drag_pad_orig_xy = None
            self._press_view = None
            self._press_canvas = None
            return

        # Region creation: commit if the rectangle is big enough.
        if self._region_create_start is not None:
            s = self._region_create_start
            ed = self._region_create_end or s
            self._region_create_start = None
            self._region_create_end = None
            cx, cy, w, h = self._view_rect_to_top_rect(
                min(s[0], ed[0]), min(s[1], ed[1]),
                max(s[0], ed[0]), max(s[1], ed[1]))
            if w >= REGION_MIN and h >= REGION_MIN:
                self.on_place_region(cx, cy, w, h, self._detect_side(),
                                     self._pending_region_color)
            self.schedule_draw()
            self._press_view = None
            self._press_canvas = None
            return

        # Pan release (spacebar drag, or any other path that set _pan_anchor).
        if self._pan_anchor is not None:
            self._pan_anchor = None
            self._press_view = None
            self._press_canvas = None
            set_canvas_cursor(
                self.canvas,
                PAN_CURSORS_OPEN if self.space_held else DEFAULT_CANVAS_CURSOR)
            return

        # Long-press: pad placement on a clean hold-and-release.
        if self._long_press.ready and self._press_view is not None:
            vx, vy = self._press_view
            tx, ty = self.view_to_top(vx, vy)
            self.on_place_pad(tx, ty, self._detect_side(), self._pending_pad_color)
        self._long_press.cancel()
        self._press_view = None
        self._press_canvas = None

    def _update_region_resize(self, e) -> None:
        idx = self._region_resize_idx
        if (idx is None or self._region_resize_handle is None
                or self._region_resize_start_rect is None
                or idx >= len(self.regions)):
            return
        rx0, ry0, rx1, ry1 = self._region_resize_start_rect
        vx, vy = self.canvas_to_orig(e.x, e.y)
        h = self._region_resize_handle
        if "n" in h: ry0 = vy
        if "s" in h: ry1 = vy
        if "w" in h: rx0 = vx
        if "e" in h: rx1 = vx
        rx0, rx1 = sorted((rx0, rx1))
        ry0, ry1 = sorted((ry0, ry1))
        cx, cy, tw, th = self._view_rect_to_top_rect(rx0, ry0, rx1, ry1)
        region = self.regions[idx]
        region.x = int(round(cx))
        region.y = int(round(cy))
        region.w = max(REGION_MIN, min(REGION_MAX, int(round(tw))))
        region.h = max(REGION_MIN, min(REGION_MAX, int(round(th))))
        self.schedule_draw()

    def _on_wheel(self, e) -> None:
        # Pads no longer use drag+scroll; plain wheel always zooms.
        super()._on_wheel(e)

    def _on_alt_wheel(self, e) -> None:
        """Alt+scroll resizes the pad under the cursor (else falls back to zoom)."""
        if self._content_size() is None:
            return
        vx, vy = self.canvas_to_orig(e.x, e.y)
        pad_idx = self._hit_pad(vx, vy)
        if pad_idx is None:
            super()._on_wheel(e)
            return
        sign = _wheel_sign(e)
        if sign == 0:
            return
        pad = self.pads[pad_idx]
        step = max(1, pad.r // 10)
        new_r = max(RADIUS_MIN, min(RADIUS_MAX, pad.r + sign * step))
        if new_r != pad.r:
            pad.r = new_r
            self.on_resize_pad(pad_idx, new_r)
            self.schedule_draw()

    # -- hover ---------------------------------------------------------------

    def _on_hover(self, e) -> None:
        if self._content_size() is None:
            self._tooltip.hide(self.schedule_draw); self._hover_idx = None
            return
        gesturing = (
            (e.state & _BUTTON_MASK)
            or self._drag_pad is not None
            or self._drag_region is not None
            or self._region_resize_handle is not None
            or self._region_create_start is not None
            or self._pan_anchor is not None
        )
        if gesturing:
            self._tooltip.hide(self.schedule_draw); self._hover_idx = None
            return
        vx, vy = self.canvas_to_orig(e.x, e.y)
        pad_hit = self._hit_pad(vx, vy)
        if pad_hit is not None:
            key = ("p", pad_hit)
            if key == self._hover_idx:
                return
            self._hover_idx = key
            pad = self.pads[pad_hit]
            title = pad.name or f"Pad #{pad_hit + 1}"
            text = f"{title}\n{pad.description}" if pad.description else title
            self._tooltip.schedule(e.x, e.y, text, on_show=self.schedule_draw)
            return
        region_hit = self._hit_region_body(vx, vy)
        if region_hit is not None:
            key = ("r", region_hit)
            if key == self._hover_idx:
                return
            self._hover_idx = key
            reg = self.regions[region_hit]
            title = reg.name or f"Region #{region_hit + 1}"
            text = f"{title}\n{reg.description}" if reg.description else title
            self._tooltip.schedule(e.x, e.y, text, on_show=self.schedule_draw)
            return
        if self._hover_idx is not None:
            self._tooltip.hide(self.schedule_draw)
            self._hover_idx = None

    def _on_leave(self) -> None:
        self._tooltip.hide(self.schedule_draw)
        self._hover_idx = None

    def _pan_start(self, e) -> None:
        super()._pan_start(e)
        self._tooltip.hide(self.schedule_draw)

    # -- drawing -------------------------------------------------------------

    def _draw_empty(self, cw: int, ch: int) -> None:
        self.canvas.create_text(cw // 2, ch // 2, fill="#bbb",
                                text="Add ≥3 paired points and press Align.")

    def _draw_content(self, ix0: int, iy0: int, ix1: int, iy1: int) -> None:
        new_w = max(1, int(round((ix1 - ix0) * self.zoom)))
        new_h = max(1, int(round((iy1 - iy0) * self.zoom)))
        crop_t, crop_w = self._scaled_crops(ix0, iy0, ix1, iy1, new_w, new_h)
        if self.single_source is None:
            blend = Image.blend(crop_t, crop_w, self.alpha)
        else:
            blend = crop_t  # safe: _composite_overlays returns a fresh image
        if self.pads or self.regions:
            blend = self._composite_overlays(blend, ix0, iy0, new_w, new_h)
        self._photo = ImageTk.PhotoImage(blend)
        cx0 = (ix0 - self.ox) * self.zoom
        cy0 = (iy0 - self.oy) * self.zoom
        self.canvas.create_image(cx0, cy0, anchor="nw", image=self._photo)

    def _scaled_crops(self, ix0: int, iy0: int, ix1: int, iy1: int,
                      new_w: int, new_h: int) -> tuple[Image.Image, Image.Image | None]:
        """Cropped + resized view-space images, cached so opacity drags are cheap.

        For single-source views the second tuple element is None; for blended
        views both crops are returned at the canvas-target size (new_w, new_h).
        """
        key = (ix0, iy0, ix1, iy1, new_w, new_h,
               id(self._top_view), id(self._warped_view), self.single_source)
        if key == self._scaled_cache_key:
            return self._scaled_cache_top, self._scaled_cache_warped

        if self.single_source is None:
            crop_t = self._top_view.crop((ix0, iy0, ix1, iy1))
            crop_w = self._warped_view.crop((ix0, iy0, ix1, iy1))
            if (new_w, new_h) != crop_t.size:
                crop_t = crop_t.resize((new_w, new_h), BILINEAR)
                crop_w = crop_w.resize((new_w, new_h), BILINEAR)
        else:
            src = self._top_view if self.single_source == "top" else self._warped_view
            crop_t = src.crop((ix0, iy0, ix1, iy1))
            if (new_w, new_h) != crop_t.size:
                crop_t = crop_t.resize((new_w, new_h), BILINEAR)
            crop_w = None

        self._scaled_cache_key = key
        self._scaled_cache_top = crop_t
        self._scaled_cache_warped = crop_w
        return crop_t, crop_w

    def _composite_overlays(self, blend: Image.Image,
                            ix0: int, iy0: int,
                            new_w: int, new_h: int) -> Image.Image:
        layer: Image.Image | None = None
        draw: ImageDraw.ImageDraw | None = None

        def ensure_layer():
            nonlocal layer, draw
            if layer is None:
                layer = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(layer)
            return draw

        # Regions first so pads paint on top.
        for reg in self.regions:
            vis = self._side_visibility(reg.side)
            if vis <= 0.0:
                continue
            rx0, ry0, rx1, ry1 = self._region_view_rect(reg)
            cx0 = (rx0 - ix0) * self.zoom
            cy0 = (ry0 - iy0) * self.zoom
            cx1 = (rx1 - ix0) * self.zoom
            cy1 = (ry1 - iy0) * self.zoom
            if cx1 < 0 or cy1 < 0 or cx0 > new_w or cy0 > new_h:
                continue
            a = int(round(max(0.0, min(1.0, reg.opacity * vis)) * 255))
            ensure_layer().rectangle(
                [cx0, cy0, cx1, cy1],
                fill=hex_to_rgb(reg.color) + (a,))

        for p in self.pads:
            vis = self._side_visibility(p.side)
            if vis <= 0.0:
                continue
            vx, vy = self.top_to_view(p.x, p.y)
            cx = (vx - ix0) * self.zoom
            cy = (vy - iy0) * self.zoom
            r = p.r * self.zoom
            if cx + r < 0 or cy + r < 0 or cx - r > new_w or cy - r > new_h:
                continue
            a = int(round(max(0.0, min(1.0, p.opacity * vis)) * 255))
            ensure_layer().ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=hex_to_rgb(p.color) + (a,))

        if layer is None:
            return blend
        return Image.alpha_composite(blend.convert("RGBA"), layer).convert("RGB")

    def _draw_overlays(self) -> None:
        c = self.canvas

        # Regions (drawn under pads so a pad on top of a region stays readable).
        for i, reg in enumerate(self.regions):
            vis = self._side_visibility(reg.side)
            if vis < 0.05:
                continue
            color = self._fade_color(reg.color, vis)
            rx0, ry0, rx1, ry1 = self._region_view_rect(reg)
            X0 = (rx0 - self.ox) * self.zoom
            Y0 = (ry0 - self.oy) * self.zoom
            X1 = (rx1 - self.ox) * self.zoom
            Y1 = (ry1 - self.oy) * self.zoom
            is_sel = i == self.selected_region
            c.create_rectangle(X0, Y0, X1, Y1, outline=color,
                               width=3 if is_sel else 2)
            if is_sel:
                halo = self._fade_color("#ffffff", vis)
                c.create_rectangle(X0 - 4, Y0 - 4, X1 + 4, Y1 + 4,
                                   outline=halo, width=1)
                mx = (X0 + X1) / 2.0
                my = (Y0 + Y1) / 2.0
                hr = self.HANDLE_R
                for hx, hy in (
                    (X0, Y0), (mx, Y0), (X1, Y0),
                    (X0, my),           (X1, my),
                    (X0, Y1), (mx, Y1), (X1, Y1),
                ):
                    c.create_oval(hx - hr, hy - hr, hx + hr, hy + hr,
                                  fill=halo, outline=color, width=2)
            label = reg.name or f"#{reg.number or i + 1}"
            c.create_text(X1 + 6, Y0, text=label, fill=color, anchor="nw",
                          font=("TkDefaultFont", 10, "bold"))

        # In-progress region creation: dashed preview.
        if self._region_create_start is not None and self._region_create_end is not None:
            sx, sy = self._region_create_start
            ex, ey = self._region_create_end
            X0 = (min(sx, ex) - self.ox) * self.zoom
            Y0 = (min(sy, ey) - self.oy) * self.zoom
            X1 = (max(sx, ex) - self.ox) * self.zoom
            Y1 = (max(sy, ey) - self.oy) * self.zoom
            c.create_rectangle(X0, Y0, X1, Y1, outline=self._pending_region_color,
                               width=2, dash=(4, 3))

        for i, p in enumerate(self.pads):
            vis = self._side_visibility(p.side)
            if vis < 0.05:
                continue
            color = self._fade_color(p.color, vis)
            vx, vy = self.top_to_view(p.x, p.y)
            X = (vx - self.ox) * self.zoom
            Y = (vy - self.oy) * self.zoom
            R = max(3.0, p.r * self.zoom)
            is_in_set = i in self.selected_pads
            is_primary = i == self.selected_pad
            c.create_oval(X - R, Y - R, X + R, Y + R,
                          outline=color, width=3 if is_in_set else 2)
            if is_in_set:
                halo = self._fade_color("#ffffff", vis)
                c.create_oval(X - R - 4, Y - R - 4, X + R + 4, Y + R + 4,
                              outline=halo, width=1)
                # Crosshair only on the single primary so the editor target
                # stays visually distinct in a multi-selection.
                if is_primary:
                    c.create_line(X - R - 6, Y, X + R + 6, Y, fill=halo)
                    c.create_line(X, Y - R - 6, X, Y + R + 6, fill=halo)
            label = p.name or f"#{p.number or i + 1}"
            c.create_text(X + R + 6, Y, text=label, fill=color, anchor="w",
                          font=("TkDefaultFont", 10, "bold"))
        self._tooltip.draw_into()
