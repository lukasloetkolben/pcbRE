"""Canvas widgets: zoomable base, alignment Panel, pad-aware OverlayView."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

from .imageops import BILINEAR, ROTATE_90, ROTATE_180, ROTATE_270
from .model import Pad, Point, Side, hex_to_rgb, random_pad_color

PANEL_W, PANEL_H = 640, 720
OVERLAY_W, OVERLAY_H = 1300, 760
WHEEL_FACTOR = 1.18
MIN_ZOOM = 0.05
MAX_ZOOM = 80.0
RADIUS_MIN = 2
RADIUS_MAX = 120

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
        sx, sy = self.canvas_to_orig(e.x, e.y)
        self.zoom = float(np.clip(self.zoom * factor, MIN_ZOOM, MAX_ZOOM))
        self.ox = sx - e.x / self.zoom
        self.oy = sy - e.y / self.zoom
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

    def __init__(self, canvas: tk.Canvas) -> None:
        self.canvas = canvas
        self.active = False
        self.ready = False
        self.color = "#ffffff"
        self._cx = 0
        self._cy = 0
        self._step = 0
        self._tick_id: str | None = None

    def start(self, cx: int, cy: int, color: str = "#ffffff") -> None:
        self.cancel()
        self.active = True
        self.ready = False
        self.color = color
        self._cx, self._cy = cx, cy
        self._step = 0
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
        r = 6 + progress * 18
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
    CLICK_THRESHOLD = 3

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
        else:
            self._drag_idx = None
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
    CLICK_THRESHOLD = 3

    def __init__(self, parent: tk.Widget,
                 on_place_pad:        Callable[[float, float, Side, str], None] | None = None,
                 on_grab_pad:         Callable[[int], None] | None = None,
                 on_move_pad:         Callable[[int, float, float], None] | None = None,
                 on_drop_pad:         Callable[[int], None] | None = None,
                 on_resize_pad:       Callable[[int, int], None] | None = None,
                 on_pad_deselect:     Callable[[], None] | None = None,
                 on_double_click_pad: Callable[[int], None] | None = None,
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

        # Pads live in TOP-image coordinates so they survive view rotation/flip.
        self.pads: list[Pad] = []
        self.selected_pad: int | None = None
        self.on_place_pad = on_place_pad or (lambda x, y, s, c: None)
        self.on_grab_pad = on_grab_pad or (lambda i: None)
        self.on_move_pad = on_move_pad or (lambda i, x, y: None)
        self.on_drop_pad = on_drop_pad or (lambda i: None)
        self.on_resize_pad = on_resize_pad or (lambda i, r: None)
        self.on_pad_deselect = on_pad_deselect or (lambda: None)
        self.on_double_click_pad = on_double_click_pad or (lambda i: None)

        self._press_view: tuple[float, float] | None = None
        self._drag_pad: int | None = None
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
        for ev in ("<ButtonPress-2>", "<ButtonPress-3>", "<Shift-ButtonPress-1>"):
            c.bind(ev, self._pan_start)
        for ev in ("<B2-Motion>", "<B3-Motion>", "<Shift-B1-Motion>"):
            c.bind(ev, self._pan_drag)

    def _on_double_click(self, e) -> None:
        if self._content_size() is None:
            self.fit()
            return
        vx, vy = self.canvas_to_orig(e.x, e.y)
        hit = self._hit_pad(vx, vy)
        if hit is not None:
            self._tooltip.hide(self.schedule_draw)
            self.on_double_click_pad(hit)
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

    def set_pair(self, top: Image.Image | None,
                 warped: Image.Image | None) -> None:
        self.top = top
        self.warped = warped
        self._rebuild_views()
        self._fitted_once = False
        self.canvas.update_idletasks()
        self.fit()

    def set_alpha(self, a: float) -> None:
        self.alpha = float(a)
        self.schedule_draw()

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

    # -- input ---------------------------------------------------------------

    def _hit_pad(self, vx: float, vy: float) -> int | None:
        if not self.pads:
            return None
        slop = 4.0 / max(self.zoom, 1e-6)
        for i in range(len(self.pads) - 1, -1, -1):
            p = self.pads[i]
            pvx, pvy = self.top_to_view(p.x, p.y)
            if np.hypot(vx - pvx, vy - pvy) <= p.r + slop:
                return i
        return None

    def _on_press(self, e) -> None:
        if self._content_size() is None:
            return
        vx, vy = self.canvas_to_orig(e.x, e.y)
        hit = self._hit_pad(vx, vy)
        self._click_moved = False
        self._tooltip.hide(self.schedule_draw)
        if hit is not None:
            self._drag_pad = hit
            self._press_view = None
            self.on_grab_pad(hit)
        else:
            self._drag_pad = None
            if self.selected_pad is not None:
                self.selected_pad = None
                self.on_pad_deselect()
            self._press_view = (vx, vy)
            self._pan_anchor = (e.x, e.y, self.ox, self.oy)
            self._pending_pad_color = random_pad_color()
            self._long_press.start(e.x, e.y, color=self._pending_pad_color)

    def _on_motion(self, e) -> None:
        if self._content_size() is None:
            return
        if self._drag_pad is not None:
            self._click_moved = True
            vx, vy = self.canvas_to_orig(e.x, e.y)
            tx, ty = self.view_to_top(vx, vy)
            self.on_move_pad(self._drag_pad, tx, ty)
            return
        if self._pan_anchor is None:
            return
        x0, y0, _, _ = self._pan_anchor
        if not self._click_moved:
            if abs(e.x - x0) <= self.CLICK_THRESHOLD and abs(e.y - y0) <= self.CLICK_THRESHOLD:
                return
            self._click_moved = True
            self._long_press.cancel()
        self._pan_drag(e)

    def _on_release(self, e) -> None:
        place_args = None
        if (self._long_press.ready
                and self._press_view is not None and self._drag_pad is None):
            vx, vy = self._press_view
            tx, ty = self.view_to_top(vx, vy)
            place_args = (tx, ty, self._detect_side(), self._pending_pad_color)
        self._long_press.cancel()
        if self._drag_pad is not None:
            if self._click_moved:
                self.on_drop_pad(self._drag_pad)
            self._drag_pad = None
        elif place_args is not None:
            self.on_place_pad(*place_args)
        self._press_view = None
        self._pan_anchor = None

    def _on_wheel(self, e) -> None:
        # Hold a pad + scroll → resize that pad instead of zooming.
        if self._drag_pad is not None and self._drag_pad < len(self.pads):
            sign = _wheel_sign(e)
            if sign == 0:
                return
            pad = self.pads[self._drag_pad]
            step = max(1, pad.r // 10)
            new_r = max(RADIUS_MIN, min(RADIUS_MAX, pad.r + sign * step))
            if new_r != pad.r:
                pad.r = new_r
                self.on_resize_pad(self._drag_pad, new_r)
                self.schedule_draw()
            return
        super()._on_wheel(e)

    # -- hover ---------------------------------------------------------------

    def _on_hover(self, e) -> None:
        if self._content_size() is None:
            self._tooltip.hide(self.schedule_draw); self._hover_idx = None
            return
        if (e.state & _BUTTON_MASK) or self._drag_pad is not None or self._pan_anchor is not None:
            self._tooltip.hide(self.schedule_draw); self._hover_idx = None
            return
        vx, vy = self.canvas_to_orig(e.x, e.y)
        hit = self._hit_pad(vx, vy)
        if hit == self._hover_idx:
            return
        self._hover_idx = hit
        if hit is None:
            self._tooltip.hide(self.schedule_draw)
            return
        pad = self.pads[hit]
        title = pad.name or f"Pad #{hit + 1}"
        text = f"{title}\n{pad.description}" if pad.description else title
        self._tooltip.schedule(e.x, e.y, text, on_show=self.schedule_draw)

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
        if self.single_source is None:
            crop_t = self._top_view.crop((ix0, iy0, ix1, iy1))
            crop_w = self._warped_view.crop((ix0, iy0, ix1, iy1))
            if (new_w, new_h) != crop_t.size:
                crop_t = crop_t.resize((new_w, new_h), BILINEAR)
                crop_w = crop_w.resize((new_w, new_h), BILINEAR)
            blend = Image.blend(crop_t, crop_w, self.alpha)
        else:
            src = self._top_view if self.single_source == "top" else self._warped_view
            blend = src.crop((ix0, iy0, ix1, iy1))
            if (new_w, new_h) != blend.size:
                blend = blend.resize((new_w, new_h), BILINEAR)
        if self.pads:
            blend = self._composite_pad_fills(blend, ix0, iy0, new_w, new_h)
        self._photo = ImageTk.PhotoImage(blend)
        cx0 = (ix0 - self.ox) * self.zoom
        cy0 = (iy0 - self.oy) * self.zoom
        self.canvas.create_image(cx0, cy0, anchor="nw", image=self._photo)

    def _composite_pad_fills(self, blend: Image.Image,
                             ix0: int, iy0: int,
                             new_w: int, new_h: int) -> Image.Image:
        layer: Image.Image | None = None
        draw: ImageDraw.ImageDraw | None = None
        for p in self.pads:
            vx, vy = self.top_to_view(p.x, p.y)
            cx = (vx - ix0) * self.zoom
            cy = (vy - iy0) * self.zoom
            r = p.r * self.zoom
            if cx + r < 0 or cy + r < 0 or cx - r > new_w or cy - r > new_h:
                continue
            if layer is None:
                layer = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(layer)
            a = int(round(max(0.0, min(1.0, p.opacity)) * 255))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=hex_to_rgb(p.color) + (a,))
        if layer is None:
            return blend
        return Image.alpha_composite(blend.convert("RGBA"), layer).convert("RGB")

    def _draw_overlays(self) -> None:
        c = self.canvas
        for i, p in enumerate(self.pads):
            vx, vy = self.top_to_view(p.x, p.y)
            X = (vx - self.ox) * self.zoom
            Y = (vy - self.oy) * self.zoom
            R = max(3.0, p.r * self.zoom)
            c.create_oval(X - R, Y - R, X + R, Y + R,
                          outline=p.color, width=3 if i == self.selected_pad else 2)
            if i == self.selected_pad:
                c.create_oval(X - R - 4, Y - R - 4, X + R + 4, Y + R + 4,
                              outline="white", width=1)
                c.create_line(X - R - 6, Y, X + R + 6, Y, fill="white")
                c.create_line(X, Y - R - 6, X, Y + R + 6, fill="white")
            label = p.name or f"#{i + 1}"
            c.create_text(X + R + 6, Y, text=label, fill=p.color, anchor="w",
                          font=("TkDefaultFont", 10, "bold"))
        self._tooltip.draw_into()
