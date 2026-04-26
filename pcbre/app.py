"""pcbRE — top-level application: toolbars, project I/O, view orchestration."""

from __future__ import annotations

import json
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps

from .imageops import fit_transform, transform_residual, warp_image
from .model import (
    PROJECT_EXT,
    PROJECT_VERSION,
    Pad,
    Point,
    Region,
    Side,
    normalize_project_data,
    normalize_side,
)
from .views import (
    OverlayView,
    Panel,
    RADIUS_MAX,
    RADIUS_MIN,
    REGION_MAX,
    REGION_MIN,
)

DEFAULT_RADIUS = 8

ALIGN_READY_BG = "#34c759"
ALIGN_READY_FG = "white"
ALIGN_DISABLED_BG = "#d1d1d6"
ALIGN_DISABLED_FG = "#6c6c70"

_DELETE_KEY_HINT = "Backspace" if sys.platform == "darwin" else "Del"
_MOD_KEY = "Cmd" if sys.platform == "darwin" else "Ctrl"
_DEFAULT_SHORTCUTS = (
    f"{_MOD_KEY}+= / {_MOD_KEY}+-  zoom    ·    "
    f"{_MOD_KEY}+0  fit    ·    "
    f"{_MOD_KEY}+S  save    ·    "
    f"{_MOD_KEY}+O  open"
)


class App:
    """Top-level controller. Holds models, owns widgets, routes events."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("pcbRE")

        self.top_path: str | None = None
        self.bottom_path: str | None = None
        self.bottom_orig: Image.Image | None = None
        self.project_path: str | None = None
        self._mirror_state = True
        self._suppress_mirror = False

        self.transform: np.ndarray | None = None
        self.warped_bottom: Image.Image | None = None
        # Single-image mode = no bottom photo, just label on the top image.
        self.single_image_mode = False

        self.bottom_mirror = tk.BooleanVar(value=True)
        self.opacity = tk.DoubleVar(value=0.5)
        self.radius = tk.DoubleVar(value=DEFAULT_RADIUS)
        # New projects open straight into overlay once alignment is in place.
        self.mode = tk.StringVar(value="overlay")
        # Place tool: "pad" (long-press to drop) or "region" (drag to box).
        self.tool_mode = tk.StringVar(value="pad")

        self.selected: tuple[Side, int] | None = None
        self._suppress_radius = False

        self._last_pad_radius = 12
        self._last_pad_opacity = 0.3
        self._pad_editor_geometry: str | None = None
        self._last_region_opacity = 0.3
        self._region_editor_geometry: str | None = None

        self._build_ui()

    # ----------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self._build_top_bars()
        self._build_status_area()
        self._build_bottom_bar()
        self._build_canvas_area()
        self._bind_keys()
        self._refresh_mode_bars()
        self._update_shortcut_bar()
        self._refresh_align_button_style()

    def _build_top_bars(self) -> None:
        common = ttk.Frame(self.root)
        common.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(common, text="Open Project", command=self.open_project).pack(side="left")
        ttk.Button(common, text="Save Project", command=self.save_project).pack(side="left", padx=(4, 0))

        self.mode_holder = ttk.Frame(self.root)
        self.mode_holder.pack(fill="x", padx=6, pady=(4, 0))

        self.align_bar = ttk.Frame(self.mode_holder)
        a = self.align_bar
        ttk.Button(a, text="Load Top",     command=self.load_top).pack(side="left")
        ttk.Button(a, text="Load Bottom",  command=self.load_bottom).pack(side="left", padx=(4, 0))
        ttk.Checkbutton(a, text="Mirror Bottom", variable=self.bottom_mirror,
                        command=self.on_mirror_toggle).pack(side="left", padx=10)
        ttk.Button(a, text="Clear All Points", command=self.clear_points).pack(side="left")
        ttk.Button(a, text="Fit View",        command=self.fit_views).pack(side="left", padx=(4, 0))
        ttk.Label(a, text="Radius:").pack(side="left", padx=(16, 4))
        self.radius_scale = ttk.Scale(
            a, from_=RADIUS_MIN, to=RADIUS_MAX, variable=self.radius,
            orient="horizontal", length=180,
            command=lambda *_: self.on_radius_change())
        self.radius_scale.pack(side="left")
        self.radius_scale.state(["disabled"])

        self.op_bar = ttk.Frame(self.mode_holder)
        o = self.op_bar
        self.reset_btn = ttk.Button(o, text="Reset Alignment", command=self.reset_alignment)
        self.reset_btn.pack(side="left")
        ttk.Separator(o, orient="vertical").pack(side="left", fill="y", padx=10)

        self.view_switcher = ttk.Frame(o)
        self.view_switcher.pack(side="left")
        ttk.Label(self.view_switcher, text="View:").pack(side="left", padx=(0, 4))
        for label, value in (("Overlay", "overlay"), ("Side-by-side", "add")):
            ttk.Radiobutton(
                self.view_switcher, text=label, value=value, variable=self.mode,
                style="Toolbutton",
                command=lambda v=value: self.set_mode(v),
            ).pack(side="left")
        self.view_switcher_sep = ttk.Separator(o, orient="vertical")
        self.view_switcher_sep.pack(side="left", fill="y", padx=10)

        ttk.Button(o, text="Fit View",     command=self.fit_views).pack(side="left")
        ttk.Button(o, text="Rotate 90°",   command=self.rotate_overlay).pack(side="left", padx=(4, 0))
        ttk.Button(o, text="Flip",         command=self.flip_overlay).pack(side="left", padx=(4, 0))
        ttk.Separator(o, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(o, text="Opacity:").pack(side="left", padx=(0, 4))
        ttk.Scale(o, from_=0.0, to=1.0, variable=self.opacity, orient="horizontal",
                  length=200, command=lambda *_: self.on_opacity_change()).pack(side="left")
        ttk.Separator(o, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(o, text="Place:").pack(side="left", padx=(0, 4))
        for label, value in (("Pad", "pad"), ("Region", "region")):
            ttk.Radiobutton(
                o, text=label, value=value, variable=self.tool_mode,
                style="Toolbutton",
                command=self._on_tool_mode_change,
            ).pack(side="left")

    def _build_status_area(self) -> None:
        self.status = ttk.Label(self.root, text="Load a top and bottom image to begin.")
        self.status.pack(fill="x", padx=8, pady=(4, 0))
        self.hint = ttk.Label(self.root, foreground="#888", text="")
        self.hint.pack(fill="x", padx=8)

    def _build_bottom_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(side="bottom", fill="x")
        self.shortcut_bar = ttk.Label(bar, foreground="#888", text="")
        self.shortcut_bar.pack(side="left")
        # Frame+Label so the bg color fills the whole button on macOS (tk.Button
        # ignores `bg` under Aqua; even tk.Label can leave a grey margin from
        # the focus highlight unless highlightthickness=0).
        self.align_btn = tk.Frame(
            bar, bg=ALIGN_DISABLED_BG,
            bd=0, highlightthickness=0,
        )
        self._align_btn_label = tk.Label(
            self.align_btn, text="Align",
            bg=ALIGN_DISABLED_BG, fg=ALIGN_DISABLED_FG,
            bd=0, highlightthickness=0,
            width=14, pady=4,
            font=("TkDefaultFont", 10, "bold"),
        )
        self._align_btn_label.pack()
        self.align_btn._enabled = True  # tracked manually since Label has no state
        for w in (self.align_btn, self._align_btn_label):
            w.bind("<Button-1>", lambda e: self._on_align_click())
        self.align_btn.pack(side="right")
        self.skip_align_btn = ttk.Button(
            bar, text="Skip alignment (top only)",
            command=self.enter_single_image_mode)
        self.skip_align_btn.pack(side="right", padx=(0, 8))

    def _build_canvas_area(self) -> None:
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.panel_top = Panel(self.canvas_frame, "top",
                               self.on_place, self.on_grab, self.on_move,
                               self.on_drop, self.on_resize_point)
        self.panel_bottom = Panel(self.canvas_frame, "bottom",
                                  self.on_place, self.on_grab, self.on_move,
                                  self.on_drop, self.on_resize_point)
        self.panel_top.pack(side="left", padx=4, fill="both", expand=True)
        self.panel_bottom.pack(side="left", padx=4, fill="both", expand=True)

        cb = dict(
            on_place_pad=self.on_place_pad,
            on_grab_pad=self.on_grab_pad,
            on_move_pad=self.on_move_pad,
            on_drop_pad=lambda idx: None,
            on_resize_pad=self.on_resize_pad,
            on_pad_deselect=self.on_pad_deselect,
            on_double_click_pad=self._open_pad_editor,
            on_place_region=self.on_place_region,
            on_grab_region=self.on_grab_region,
            on_move_region=self.on_move_region,
            on_resize_region=self.on_resize_region,
            on_region_deselect=self.on_region_deselect,
            on_double_click_region=self._open_region_editor,
        )
        self.overlay = OverlayView(self.canvas_frame, **cb)
        # Side-by-side wrapper: grid with two uniform columns guarantees both
        # halves are always exactly (parent - margins) / 2 wide.
        self.sbs_frame = ttk.Frame(self.canvas_frame)
        self.sbs_frame.columnconfigure(0, weight=1, uniform="sbs")
        self.sbs_frame.columnconfigure(1, weight=1, uniform="sbs")
        self.sbs_frame.rowconfigure(0, weight=1)
        self.overlay_left = OverlayView(self.sbs_frame, single_source="top", **cb)
        self.overlay_right = OverlayView(self.sbs_frame, single_source="warped", **cb)
        # All three views share the same pads/regions lists (mutate in place — never replace).
        self.overlay_left.pads = self.overlay.pads
        self.overlay_right.pads = self.overlay.pads
        self.overlay_left.regions = self.overlay.regions
        self.overlay_right.regions = self.overlay.regions
        self._set_last_pad_radius(self._last_pad_radius)
        self._on_tool_mode_change()

    def _bind_keys(self) -> None:
        for seq, fn in [
            ("<Control-s>", self.save_project), ("<Command-s>", self.save_project),
            ("<Control-Shift-S>", self.save_project_as),
            ("<Command-Shift-S>", self.save_project_as),
            ("<Control-o>", self.open_project), ("<Command-o>", self.open_project),
        ]:
            self.root.bind(seq, lambda e, f=fn: f())
        self.root.bind("<KeyPress-e>", self._on_e_key)
        self.root.bind("<KeyPress-E>", self._on_e_key)
        for seq in ("<Delete>", "<BackSpace>", "<KP_Delete>"):
            self.root.bind(seq, self._on_delete_key)
        # Cmd/Ctrl + = / + / - / 0 → zoom in / zoom out / fit, centered on the
        # canvas under the cursor (or the active overlay's center as fallback).
        for seq in ("<Command-equal>", "<Command-plus>",
                    "<Control-equal>", "<Control-plus>"):
            self.root.bind(seq, lambda e: self._kb_zoom(1.18))
        for seq in ("<Command-minus>", "<Control-minus>"):
            self.root.bind(seq, lambda e: self._kb_zoom(1 / 1.18))
        for seq in ("<Command-0>", "<Control-0>"):
            self.root.bind(seq, lambda e: self.fit_views())

    def _kb_zoom(self, factor: float) -> None:
        try:
            x = self.root.winfo_pointerx()
            y = self.root.winfo_pointery()
            target = self.root.winfo_containing(x, y)
        except tk.TclError:
            target = None
        views = (self.panel_top, self.panel_bottom,
                 self.overlay, self.overlay_left, self.overlay_right)
        for v in views:
            if target is v.canvas:
                cx = x - v.canvas.winfo_rootx()
                cy = y - v.canvas.winfo_rooty()
                v._zoom_at(cx, cy, factor)
                return
        for v in views:
            try:
                if v.canvas.winfo_ismapped():
                    v._zoom_at(v.canvas.winfo_width() // 2,
                               v.canvas.winfo_height() // 2, factor)
                    return
            except tk.TclError:
                pass

    # ---------------------------------------------------------- mode bars

    def _refresh_mode_bars(self) -> None:
        aligned = self._is_aligned()
        if aligned:
            self.align_bar.pack_forget()
            if not self.op_bar.winfo_ismapped():
                self.op_bar.pack(fill="x")
            self.hint.config(text=(
                "scroll = zoom · right/middle/Shift+drag = pan · "
                "double-click = fit · hold ~0.3 s = pad · "
                "Region tool: drag a box · Alt+scroll on pad = resize · "
                "drag a dot on the selected region to resize"))
            # Side-by-side view only makes sense with two images.
            if self.single_image_mode:
                self.view_switcher.pack_forget()
                self.view_switcher_sep.pack_forget()
                if self.mode.get() != "overlay":
                    self.mode.set("overlay")
            else:
                if not self.view_switcher.winfo_ismapped():
                    self.view_switcher.pack(side="left",
                                            before=self.view_switcher_sep)
                if not self.view_switcher_sep.winfo_ismapped():
                    self.view_switcher_sep.pack(side="left", fill="y", padx=10)
        else:
            self.op_bar.pack_forget()
            if not self.align_bar.winfo_ismapped():
                self.align_bar.pack(fill="x")
            self.hint.config(text=(
                "Alignment: click on TOP, then matching point on BOTTOM. "
                "Drag a point to move it · drag + scroll = resize · "
                "≥3 pairs to Align (or Skip alignment for top-only)"))
        self._refresh_align_button_style()
        if hasattr(self, "panel_top"):
            self.set_mode(self.mode.get())
        self._update_shortcut_bar()

    def enter_single_image_mode(self) -> None:
        if self.panel_top.image is None:
            messagebox.showinfo("No top image",
                                "Load a top image first.")
            return
        self.single_image_mode = True
        self.transform = np.eye(3)
        self.warped_bottom = self.panel_top.image
        for v in self._pad_views():
            v.set_pair(self.panel_top.image, self.panel_top.image)
            v.set_alpha(self.opacity.get())
        self.mode.set("overlay")
        self._refresh_mode_bars()
        self.status.config(text="Top-only mode — click and hold to label pads.")

    def _refresh_align_button_style(self) -> None:
        if self._is_aligned():
            self.align_btn.pack_forget()
            self.skip_align_btn.pack_forget()
            return
        if not self.align_btn.winfo_ismapped():
            self.align_btn.pack(side="right")
        nt, nb = len(self.panel_top.points), len(self.panel_bottom.points)
        ready = min(nt, nb) >= 3
        self.align_btn._enabled = ready
        bg = ALIGN_READY_BG if ready else ALIGN_DISABLED_BG
        fg = ALIGN_READY_FG if ready else ALIGN_DISABLED_FG
        cursor = "hand2" if ready else ""
        self.align_btn.config(bg=bg, cursor=cursor)
        self._align_btn_label.config(bg=bg, fg=fg, cursor=cursor)
        # Skip alignment is only meaningful with a top image and no bottom yet.
        skip_visible = self.panel_top.image is not None and self.panel_bottom.image is None
        if skip_visible and not self.skip_align_btn.winfo_ismapped():
            self.skip_align_btn.pack(side="right", padx=(0, 8))
        elif not skip_visible:
            self.skip_align_btn.pack_forget()

    # --------------------------------------------------------- bottom hint

    def _update_shortcut_bar(self) -> None:
        if self._is_aligned() and self.overlay.selected_pad is not None:
            pad = self.overlay.pads[self.overlay.selected_pad]
            label = pad.name or f"#{self.overlay.selected_pad + 1}"
            self.shortcut_bar.config(
                text=f'Selected pad: {label}   ·   "E" edit   ·   {_DELETE_KEY_HINT} delete')
        elif self._is_aligned() and self.overlay.selected_region is not None:
            reg = self.overlay.regions[self.overlay.selected_region]
            label = reg.name or f"R{self.overlay.selected_region + 1}"
            self.shortcut_bar.config(
                text=(f'Selected region: {label}   ·   "E" edit   ·   '
                      f"{_DELETE_KEY_HINT} delete   ·   drag dot to resize"))
        elif (not self._is_aligned()) and self.selected is not None:
            side, idx = self.selected
            self.shortcut_bar.config(
                text=f"Selected point #{idx + 1} ({side.upper()})   "
                     f"·   {_DELETE_KEY_HINT} delete pair")
        else:
            self.shortcut_bar.config(text=_DEFAULT_SHORTCUTS)

    def _focused_in_text_input(self) -> bool:
        focused = self.root.focus_get()
        return isinstance(focused, (tk.Entry, ttk.Entry, tk.Text))

    def _on_e_key(self, e=None) -> None:
        if self._focused_in_text_input() or not self._is_aligned():
            return
        pad_idx = self.overlay.selected_pad
        if pad_idx is not None and pad_idx < len(self.overlay.pads):
            self._open_pad_editor(pad_idx)
            return
        reg_idx = self.overlay.selected_region
        if reg_idx is not None and reg_idx < len(self.overlay.regions):
            self._open_region_editor(reg_idx)

    def _on_delete_key(self, e=None) -> None:
        if self._focused_in_text_input():
            return
        if self._is_aligned() and self.overlay.selected_pad is not None:
            self._delete_selected_pad()
            return
        if self._is_aligned() and self.overlay.selected_region is not None:
            self._delete_selected_region()
            return
        if (not self._is_aligned()) and self.selected is not None:
            self._delete_selected_point_pair()

    def _delete_selected_pad(self) -> None:
        idx = self.overlay.selected_pad
        if idx is None or idx >= len(self.overlay.pads):
            return
        del self.overlay.pads[idx]
        self._set_selected_pad(None)

    def _delete_selected_region(self) -> None:
        idx = self.overlay.selected_region
        if idx is None or idx >= len(self.overlay.regions):
            return
        del self.overlay.regions[idx]
        self._set_selected_region(None)

    def _delete_selected_point_pair(self) -> None:
        if self.selected is None:
            return
        _, idx = self.selected
        for pts in (self.panel_top.points, self.panel_bottom.points):
            if idx < len(pts):
                pts.pop(idx)
        self._select(None, None)
        self._invalidate_alignment()
        self.panel_top.draw(); self.panel_bottom.draw()
        self._update_status()

    # -------------------------------------------------------------- file I/O

    def _open_image_dialog(self, title: str) -> tuple[Image.Image, str] | None:
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return None
        try:
            return Image.open(path).convert("RGB"), path
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not open image", str(e))
            return None

    def load_top(self) -> None:
        if self._warn_locked("load a new top image"):
            return
        result = self._open_image_dialog("Select TOP image")
        if result is None:
            return
        img, path = result
        self.top_path = path
        self.panel_top.points.clear()
        self.panel_top.set_image(img)
        self._select(None, None)
        self._invalidate_alignment()
        self._update_status()

    def load_bottom(self) -> None:
        if self._warn_locked("load a new bottom image"):
            return
        result = self._open_image_dialog("Select BOTTOM image")
        if result is None:
            return
        img, path = result
        self.bottom_path = path
        self.bottom_orig = img
        self._mirror_state = self.bottom_mirror.get()
        view = ImageOps.mirror(img) if self._mirror_state else img
        self.panel_bottom.points.clear()
        self.panel_bottom.set_image(view)
        self._select(None, None)
        self._invalidate_alignment()
        self._update_status()

    def on_mirror_toggle(self) -> None:
        if self._suppress_mirror:
            return
        new_state = self.bottom_mirror.get()
        if self._is_aligned() and new_state != self._mirror_state:
            self._suppress_mirror = True
            self.bottom_mirror.set(self._mirror_state)
            self._suppress_mirror = False
            self._warn_locked("toggle mirror")
            return
        if self.bottom_orig is None or new_state == self._mirror_state:
            self._mirror_state = new_state
            return
        view = ImageOps.mirror(self.bottom_orig) if new_state else self.bottom_orig
        w = view.width
        self.panel_bottom.points = [
            Point(x=w - 1 - p.x, y=p.y, r=p.r) for p in self.panel_bottom.points
        ]
        self._mirror_state = new_state
        self.panel_bottom.image = view
        self.panel_bottom.draw()
        self._invalidate_alignment()
        self._update_status()

    # --------------------------------------------------------- project I/O

    def _set_title(self) -> None:
        base = "pcbRE"
        self.root.title(f"{base} — {Path(self.project_path).name}"
                        if self.project_path else base)

    def save_project(self) -> None:
        if self.project_path:
            self._do_save(self.project_path)
        else:
            self.save_project_as()

    def save_project_as(self) -> None:
        if self.top_path is None:
            messagebox.showinfo("Nothing to save", "Load a top image first.")
            return
        if not self.single_image_mode and self.bottom_path is None:
            messagebox.showinfo("Nothing to save",
                                "Load a bottom image or click Skip alignment first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=PROJECT_EXT,
            filetypes=[("pcbRE Project", f"*{PROJECT_EXT}")])
        if not path:
            return
        try:
            final_path = self._bundle_into_folder(path)
        except OSError as e:
            messagebox.showerror("Could not create project folder", str(e))
            return
        self._do_save(final_path)
        self.project_path = final_path
        self._set_title()

    def _bundle_into_folder(self, picked_path: str) -> str:
        """Create `<picked-name>/` next to where the user pointed, copy the
        loaded images to canonical names inside it, and return the full
        path of the project file that should live in the folder."""
        p = Path(picked_path)
        project_dir = p.with_suffix("")
        project_dir.mkdir(parents=True, exist_ok=True)

        self.top_path = self._copy_into(self.top_path, project_dir, "top")
        if self.bottom_path is not None:
            self.bottom_path = self._copy_into(self.bottom_path, project_dir, "bottom")
        return str(project_dir / p.name)

    @staticmethod
    def _copy_into(src_path: str, project_dir: Path, name: str) -> str:
        src = Path(src_path).resolve()
        ext = src.suffix.lower() or ".png"
        dst = (project_dir / f"{name}{ext}").resolve()
        if src != dst:
            shutil.copy2(src, dst)
        return str(dst)

    def _do_save(self, path: str) -> None:
        proj_dir = Path(path).resolve().parent

        def rel(p: str | None) -> str | None:
            if p is None:
                return None
            ap = Path(p).resolve()
            try:
                return str(ap.relative_to(proj_dir))
            except ValueError:
                return str(ap)

        data = {
            "version": PROJECT_VERSION,
            "images": {"top": rel(self.top_path), "bottom": rel(self.bottom_path)},
            "bottom_mirror": bool(self.bottom_mirror.get()),
            "single_image": self.single_image_mode,
            "view": {
                "rotation": int(self.overlay.rotation),
                "flipped": bool(self.overlay.flipped),
                "opacity": float(self.opacity.get()),
                "mode": self.mode.get(),
            },
            "alignment_points": {
                "top":    [self._point_dict(p) for p in self.panel_top.points],
                "bottom": [self._point_dict(p) for p in self.panel_bottom.points],
            },
            "pads": [self._pad_dict(p) for p in self.overlay.pads],
            "regions": [self._region_dict(r) for r in self.overlay.regions],
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not save project", str(e))
            return
        self.status.config(text=f"Saved project: {path}")

    @staticmethod
    def _point_dict(p: Point) -> dict:
        return {"x": p.x, "y": p.y, "r": p.r}

    @staticmethod
    def _pad_dict(p: Pad) -> dict:
        return {
            "x": p.x, "y": p.y, "r": p.r,
            "name": p.name, "description": p.description,
            "color": p.color, "opacity": p.opacity, "side": p.side,
        }

    @staticmethod
    def _region_dict(r: Region) -> dict:
        return {
            "x": r.x, "y": r.y, "w": r.w, "h": r.h,
            "name": r.name, "description": r.description,
            "color": r.color, "opacity": r.opacity, "side": r.side,
        }

    def open_project(self) -> None:
        path = filedialog.askopenfilename(
            title="Open project",
            filetypes=[("pcbRE Project", f"*{PROJECT_EXT}"),
                       ("JSON", "*.json"), ("All files", "*.*")])
        if path:
            self._do_open(path)

    def _resolve(self, ref: str | None, proj_dir: Path) -> str | None:
        if not ref:
            return None
        p = Path(ref)
        if not p.is_absolute():
            p = (proj_dir / ref).resolve()
        if p.exists():
            return str(p)
        # Last-ditch: same filename next to the project.
        alt = proj_dir / Path(ref).name
        return str(alt) if alt.exists() else None

    def _do_open(self, path: str) -> None:
        try:
            with open(path) as f:
                doc = json.load(f)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not open project", str(e))
            return
        data = normalize_project_data(doc)

        proj_dir = Path(path).resolve().parent
        top_path = self._resolve(data["images"]["top"], proj_dir)
        bottom_path = self._resolve(data["images"]["bottom"], proj_dir)
        single_image = data["single_image"] or bottom_path is None
        if top_path is None:
            messagebox.showerror("Top image not found",
                                 f"Could not locate: {data['images']['top']}")
            return
        if not single_image and bottom_path is None:
            messagebox.showerror("Bottom image not found",
                                 f"Could not locate: {data['images']['bottom']}")
            return
        try:
            top_img = Image.open(top_path).convert("RGB")
            bottom_img = (Image.open(bottom_path).convert("RGB")
                          if bottom_path else None)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not load images", str(e))
            return

        self.single_image_mode = False  # flipped on after enter_single_image_mode below
        mirror = data["bottom_mirror"]
        self.top_path = top_path
        self.bottom_path = bottom_path
        self.bottom_orig = bottom_img
        self._mirror_state = mirror
        self._suppress_mirror = True
        self.bottom_mirror.set(mirror)
        self._suppress_mirror = False

        self.panel_top.set_image(top_img)
        if bottom_img is not None:
            self.panel_bottom.set_image(ImageOps.mirror(bottom_img) if mirror else bottom_img)
        else:
            self.panel_bottom.set_image(None)

        self._load_points(data["alignment_points"])
        self._load_pads(data["pads"])
        self._load_regions(data.get("regions", []))
        self._apply_view_state(data["view"])

        self.project_path = path
        self._set_title()
        self._select(None, None)
        self._invalidate_alignment()
        self._update_shortcut_bar()
        self.panel_top.draw(); self.panel_bottom.draw()

        if single_image:
            self.enter_single_image_mode()
            self.status.config(text=f"Opened project: {path}  ·  top-only mode")
        elif min(len(self.panel_top.points), len(self.panel_bottom.points)) >= 3:
            self.align()
            self.set_mode("overlay")
            for v in self._pad_views():
                v.set_alpha(self.opacity.get())
            self.status.config(text=f"Opened project: {path}  ·  alignment restored")
        else:
            self.status.config(text=f"Opened project: {path}")
        self._update_status()

    def _load_points(self, align_pts: dict) -> None:
        try:
            self.panel_top.points = [
                Point(int(p["x"]), int(p["y"]), int(p["r"])) for p in align_pts["top"]]
            self.panel_bottom.points = [
                Point(int(p["x"]), int(p["y"]), int(p["r"])) for p in align_pts["bottom"]]
        except (KeyError, TypeError, ValueError) as e:
            messagebox.showerror("Bad point data", str(e))
            self.panel_top.points.clear(); self.panel_bottom.points.clear()

    def _load_pads(self, pads_data: list) -> None:
        # Mutate the shared list in place — overlay_left/right reference it by identity.
        try:
            new_pads = [
                Pad(x=int(p["x"]), y=int(p["y"]),
                    r=int(p.get("r", 12)),
                    name=str(p.get("name", "")),
                    description=str(p.get("description", "")),
                    color=str(p.get("color", "#ff3b30")),
                    opacity=float(p.get("opacity", 0.3)),
                    side=normalize_side(str(p.get("side", "top"))))
                for p in pads_data
            ]
        except (KeyError, TypeError, ValueError) as e:
            messagebox.showerror("Bad pad data", str(e))
            new_pads = []
        self.overlay.pads.clear()
        self.overlay.pads.extend(new_pads)
        for v in self._pad_views():
            v.selected_pad = None
        if self.overlay.pads:
            self._set_last_pad_radius(self.overlay.pads[-1].r)

    def _load_regions(self, regions_data: list) -> None:
        try:
            new_regions = [
                Region(x=int(r["x"]), y=int(r["y"]),
                       w=int(r.get("w", 80)), h=int(r.get("h", 50)),
                       name=str(r.get("name", "")),
                       description=str(r.get("description", "")),
                       color=str(r.get("color", "#0a84ff")),
                       opacity=float(r.get("opacity", 0.3)),
                       side=normalize_side(str(r.get("side", "top"))))
                for r in regions_data
            ]
        except (KeyError, TypeError, ValueError) as e:
            messagebox.showerror("Bad region data", str(e))
            new_regions = []
        self.overlay.regions.clear()
        self.overlay.regions.extend(new_regions)
        for v in self._pad_views():
            v.selected_region = None
        if self.overlay.regions:
            self._last_region_opacity = self.overlay.regions[-1].opacity

    def _apply_view_state(self, view: dict) -> None:
        for v in self._pad_views():
            v.rotation = view["rotation"]
            v.flipped = view["flipped"]
        self.opacity.set(view["opacity"])

    # ----------------------------------------------------- point callbacks

    def _panel(self, side: Side) -> Panel:
        return self.panel_top if side == "top" else self.panel_bottom

    def on_place(self, side: Side, ox: float, oy: float) -> None:
        # Panels only fire on_place during alignment setup (set_mode hides them
        # once aligned), so a lock check is enough here.
        if self._warn_locked("add points"):
            return
        panel = self._panel(side)
        if panel.image is None:
            return
        nt, nb = len(self.panel_top.points), len(self.panel_bottom.points)
        if side == "top" and nt > nb:
            self.status.config(text="Now click the matching hole on BOTTOM.")
            return
        if side == "bottom" and nb >= nt:
            self.status.config(text="Click a point on TOP first.")
            return
        r = max(RADIUS_MIN, min(RADIUS_MAX, int(round(self.radius.get()))))
        panel.points.append(Point(int(round(ox)), int(round(oy)), r))
        self._invalidate_alignment()
        self._select(side, len(panel.points) - 1)

    def on_grab(self, side: Side, idx: int) -> None:
        if self._warn_locked("move points"):
            return
        self._select(side, idx)

    def on_move(self, side: Side, idx: int, ox: float, oy: float) -> None:
        if self._is_aligned():
            return
        panel = self._panel(side)
        if idx >= len(panel.points):
            return
        pt = panel.points[idx]
        pt.x = int(round(ox)); pt.y = int(round(oy))
        panel.draw()  # alignment is invalidated on drop, not on every motion

    def on_drop(self, side: Side, idx: int) -> None:
        if self._is_aligned():
            return
        self._invalidate_alignment()
        self._select(side, idx)

    def on_resize_point(self, side: Side, idx: int, r: int) -> None:
        if self._is_aligned():
            return
        panel = self._panel(side)
        if idx >= len(panel.points):
            return
        r = max(RADIUS_MIN, min(RADIUS_MAX, int(r)))
        pt = panel.points[idx]
        if pt.r == r:
            return
        pt.r = r
        self._invalidate_alignment()
        panel.draw()
        if self.selected == (side, idx):
            self._suppress_radius = True
            self.radius.set(r)
            self._suppress_radius = False
        self._update_status()

    # ------------------------------------------------------- pad callbacks

    def on_place_pad(self, tx: float, ty: float, side: Side, color: str) -> None:
        if self.overlay.top is None:
            return
        tx = float(np.clip(tx, 0, self.overlay.top.width - 1))
        ty = float(np.clip(ty, 0, self.overlay.top.height - 1))
        r = max(RADIUS_MIN, min(RADIUS_MAX, int(self._last_pad_radius)))
        if self.single_image_mode:
            side = "top"
        pad = Pad(
            x=int(round(tx)), y=int(round(ty)),
            r=r,
            color=color,
            opacity=self._last_pad_opacity,
            side=side,
        )
        self.overlay.pads.append(pad)
        self._set_selected_pad(len(self.overlay.pads) - 1)
        self.status.config(text=f"Placed pad on {side.upper()} side — press E to edit.")

    def on_grab_pad(self, idx: int) -> None:
        self._set_selected_pad(idx)

    def on_move_pad(self, idx: int, tx: float, ty: float) -> None:
        if idx >= len(self.overlay.pads) or self.overlay.top is None:
            return
        tx = float(np.clip(tx, 0, self.overlay.top.width - 1))
        ty = float(np.clip(ty, 0, self.overlay.top.height - 1))
        pad = self.overlay.pads[idx]
        pad.x = int(round(tx)); pad.y = int(round(ty))
        self._draw_pad_views()

    def on_resize_pad(self, idx: int, r: int) -> None:
        if idx >= len(self.overlay.pads):
            return
        self._set_last_pad_radius(r)
        self._draw_pad_views()

    def _set_last_pad_radius(self, r: int) -> None:
        self._last_pad_radius = r
        for v in self._pad_views():
            v.next_pad_radius = r

    def on_pad_deselect(self) -> None:
        self._set_selected_pad(None)

    # ---------------------------------------------------- region callbacks

    def on_place_region(self, tx: float, ty: float, w: float, h: float,
                        side: Side, color: str) -> None:
        if self.overlay.top is None:
            return
        W, H = self.overlay.top.size
        tx = float(np.clip(tx, 0, W - 1))
        ty = float(np.clip(ty, 0, H - 1))
        w_i = max(REGION_MIN, min(REGION_MAX, int(round(w))))
        h_i = max(REGION_MIN, min(REGION_MAX, int(round(h))))
        if self.single_image_mode:
            side = "top"
        region = Region(
            x=int(round(tx)), y=int(round(ty)),
            w=w_i, h=h_i,
            color=color,
            opacity=self._last_region_opacity,
            side=side,
        )
        self.overlay.regions.append(region)
        self._set_selected_region(len(self.overlay.regions) - 1)
        self.status.config(text=f"Placed region on {side.upper()} side — press E to edit.")

    def on_grab_region(self, idx: int) -> None:
        self._set_selected_region(idx)

    def on_move_region(self, idx: int, tx: float, ty: float) -> None:
        if idx >= len(self.overlay.regions) or self.overlay.top is None:
            return
        W, H = self.overlay.top.size
        tx = float(np.clip(tx, 0, W - 1))
        ty = float(np.clip(ty, 0, H - 1))
        region = self.overlay.regions[idx]
        region.x = int(round(tx)); region.y = int(round(ty))
        self._draw_pad_views()

    def on_resize_region(self, idx: int, tx: float, ty: float,
                         w: float, h: float) -> None:
        if idx >= len(self.overlay.regions):
            return
        # The view already updated x/y/w/h live; this fires once on release.
        self._draw_pad_views()

    def on_region_deselect(self) -> None:
        self._set_selected_region(None)

    def _on_tool_mode_change(self) -> None:
        mode = self.tool_mode.get()
        for v in self._pad_views():
            v.set_tool_mode(mode)

    # ----------------------------------------------------- pad-view helpers

    def _pad_views(self) -> list[OverlayView]:
        return [self.overlay, self.overlay_left, self.overlay_right]

    def _set_selected_pad(self, idx: int | None) -> None:
        for v in self._pad_views():
            v.selected_pad = idx
            if idx is not None:
                v.selected_region = None
        self._update_shortcut_bar()
        self._draw_pad_views()

    def _set_selected_region(self, idx: int | None) -> None:
        for v in self._pad_views():
            v.selected_region = idx
            if idx is not None:
                v.selected_pad = None
        self._update_shortcut_bar()
        self._draw_pad_views()

    def _draw_pad_views(self) -> None:
        for v in self._pad_views():
            v.schedule_draw()

    # ----------------------------------------------------------- pad editor

    def _open_pad_editor(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.overlay.pads):
            return
        pad = self.overlay.pads[idx]

        win = tk.Toplevel(self.root)
        win.title(f"Pad #{idx + 1}")
        win.transient(self.root)
        win.minsize(420, 380)
        if self._pad_editor_geometry:
            try:
                win.geometry(self._pad_editor_geometry)
            except tk.TclError:
                pass

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(1, weight=1)

        ttk.Label(frm, text="Name:").grid(row=0, column=0, sticky="w")
        name_var = tk.StringVar(value=pad.name)
        name_entry = ttk.Entry(frm, textvariable=name_var)
        name_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)

        ttk.Label(frm, text="Description:").grid(row=1, column=0, sticky="nw", pady=(8, 0))
        desc_text = tk.Text(frm, width=40, height=8, wrap="word")
        desc_text.insert("1.0", pad.description)
        desc_text.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Color:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        color_swatch = tk.Button(frm, text="  ", width=4, relief="ridge",
                                 bg=pad.color, activebackground=pad.color)
        color_swatch.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Opacity:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        op_var = tk.DoubleVar(value=pad.opacity)
        ttk.Scale(frm, from_=0.0, to=1.0, variable=op_var, orient="horizontal",
                  length=200).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Size:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        size_row = ttk.Frame(frm)
        size_row.grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        size_var = tk.DoubleVar(value=pad.r)
        ttk.Scale(size_row, from_=RADIUS_MIN, to=RADIUS_MAX, variable=size_var,
                  orient="horizontal", length=180).pack(side="left")
        size_lbl = ttk.Label(size_row, text=f"{pad.r}px", width=6)
        size_lbl.pack(side="left", padx=(6, 0))

        ttk.Label(frm, text=f"Side: {pad.side.upper()}", foreground="#888").grid(
            row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        def pick_color():
            result = colorchooser.askcolor(color=pad.color, title="Pad color", parent=win)
            if not result or not result[1]:
                return
            pad.color = result[1]
            color_swatch.config(bg=pad.color, activebackground=pad.color)
            self._draw_pad_views()
        color_swatch.config(command=pick_color)

        def commit_name(*_):
            pad.name = name_var.get()
            self._update_shortcut_bar()
            self._draw_pad_views()

        def commit_description(_=None):
            pad.description = desc_text.get("1.0", "end-1c")
            desc_text.edit_modified(False)

        def commit_opacity(*_):
            pad.opacity = float(op_var.get())
            self._last_pad_opacity = pad.opacity
            self._draw_pad_views()

        def commit_size(*_):
            r = max(RADIUS_MIN, min(RADIUS_MAX, int(round(size_var.get()))))
            size_lbl.config(text=f"{r}px")
            if pad.r != r:
                pad.r = r
                self._set_last_pad_radius(r)
                self._draw_pad_views()

        def delete_pad():
            if pad in self.overlay.pads:
                self.overlay.pads.remove(pad)
            self.overlay.selected_pad = None
            self._update_shortcut_bar()
            self._draw_pad_views()
            win.destroy()

        def remember_geometry(_=None):
            try:
                self._pad_editor_geometry = win.geometry()
            except tk.TclError:
                pass

        name_var.trace_add("write", commit_name)
        desc_text.bind("<<Modified>>", commit_description)
        op_var.trace_add("write", commit_opacity)
        size_var.trace_add("write", commit_size)
        win.bind("<Configure>", remember_geometry)
        win.protocol("WM_DELETE_WINDOW",
                     lambda: (remember_geometry(), win.destroy()))

        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(btns, text="Delete pad", command=delete_pad).pack(side="left")
        ttk.Button(btns, text="Close",
                   command=lambda: (remember_geometry(), win.destroy())).pack(side="right")
        win.bind("<Escape>", lambda e: (remember_geometry(), win.destroy()))
        name_entry.focus_set()
        name_entry.icursor("end")

    def _open_region_editor(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.overlay.regions):
            return
        region = self.overlay.regions[idx]

        win = tk.Toplevel(self.root)
        win.title(f"Region #{idx + 1}")
        win.transient(self.root)
        win.minsize(420, 420)
        if self._region_editor_geometry:
            try:
                win.geometry(self._region_editor_geometry)
            except tk.TclError:
                pass

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(1, weight=1)

        ttk.Label(frm, text="Name:").grid(row=0, column=0, sticky="w")
        name_var = tk.StringVar(value=region.name)
        name_entry = ttk.Entry(frm, textvariable=name_var)
        name_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)

        ttk.Label(frm, text="Description:").grid(row=1, column=0, sticky="nw", pady=(8, 0))
        desc_text = tk.Text(frm, width=40, height=8, wrap="word")
        desc_text.insert("1.0", region.description)
        desc_text.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Color:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        color_swatch = tk.Button(frm, text="  ", width=4, relief="ridge",
                                 bg=region.color, activebackground=region.color)
        color_swatch.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Opacity:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        op_var = tk.DoubleVar(value=region.opacity)
        ttk.Scale(frm, from_=0.0, to=1.0, variable=op_var, orient="horizontal",
                  length=200).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Width:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        w_row = ttk.Frame(frm)
        w_row.grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        w_var = tk.DoubleVar(value=region.w)
        ttk.Scale(w_row, from_=REGION_MIN, to=400, variable=w_var,
                  orient="horizontal", length=180).pack(side="left")
        w_lbl = ttk.Label(w_row, text=f"{region.w}px", width=7)
        w_lbl.pack(side="left", padx=(6, 0))

        ttk.Label(frm, text="Height:").grid(row=5, column=0, sticky="w", pady=(8, 0))
        h_row = ttk.Frame(frm)
        h_row.grid(row=5, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        h_var = tk.DoubleVar(value=region.h)
        ttk.Scale(h_row, from_=REGION_MIN, to=400, variable=h_var,
                  orient="horizontal", length=180).pack(side="left")
        h_lbl = ttk.Label(h_row, text=f"{region.h}px", width=7)
        h_lbl.pack(side="left", padx=(6, 0))

        ttk.Label(frm, text=f"Side: {region.side.upper()}", foreground="#888").grid(
            row=6, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        def pick_color():
            result = colorchooser.askcolor(color=region.color,
                                           title="Region color", parent=win)
            if not result or not result[1]:
                return
            region.color = result[1]
            color_swatch.config(bg=region.color, activebackground=region.color)
            self._draw_pad_views()
        color_swatch.config(command=pick_color)

        def commit_name(*_):
            region.name = name_var.get()
            self._update_shortcut_bar()
            self._draw_pad_views()

        def commit_description(_=None):
            region.description = desc_text.get("1.0", "end-1c")
            desc_text.edit_modified(False)

        def commit_opacity(*_):
            region.opacity = float(op_var.get())
            self._last_region_opacity = region.opacity
            self._draw_pad_views()

        def commit_w(*_):
            v = max(REGION_MIN, min(REGION_MAX, int(round(w_var.get()))))
            w_lbl.config(text=f"{v}px")
            if region.w != v:
                region.w = v
                self._draw_pad_views()

        def commit_h(*_):
            v = max(REGION_MIN, min(REGION_MAX, int(round(h_var.get()))))
            h_lbl.config(text=f"{v}px")
            if region.h != v:
                region.h = v
                self._draw_pad_views()

        def delete_region():
            if region in self.overlay.regions:
                self.overlay.regions.remove(region)
            for v in self._pad_views():
                v.selected_region = None
            self._update_shortcut_bar()
            self._draw_pad_views()
            win.destroy()

        def remember_geometry(_=None):
            try:
                self._region_editor_geometry = win.geometry()
            except tk.TclError:
                pass

        name_var.trace_add("write", commit_name)
        desc_text.bind("<<Modified>>", commit_description)
        op_var.trace_add("write", commit_opacity)
        w_var.trace_add("write", commit_w)
        h_var.trace_add("write", commit_h)
        win.bind("<Configure>", remember_geometry)
        win.protocol("WM_DELETE_WINDOW",
                     lambda: (remember_geometry(), win.destroy()))

        btns = ttk.Frame(frm)
        btns.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(btns, text="Delete region", command=delete_region).pack(side="left")
        ttk.Button(btns, text="Close",
                   command=lambda: (remember_geometry(), win.destroy())).pack(side="right")
        win.bind("<Escape>", lambda e: (remember_geometry(), win.destroy()))
        name_entry.focus_set()
        name_entry.icursor("end")

    # -------------------------------------------------------- selection

    def _select(self, side: Side | None, idx: int | None) -> None:
        if side is None or idx is None:
            self.selected = None
            self.panel_top.selected_index = None
            self.panel_bottom.selected_index = None
            self.radius_scale.state(["disabled"])
        else:
            self.selected = (side, idx)
            sel_top = side == "top"
            self.panel_top.selected_index = idx if sel_top else None
            self.panel_bottom.selected_index = None if sel_top else idx
            pt = self._panel(side).points[idx]
            self._suppress_radius = True
            self.radius.set(pt.r)
            self._suppress_radius = False
            self.radius_scale.state(["!disabled"])
        self.panel_top.draw(); self.panel_bottom.draw()
        self._update_shortcut_bar()
        self._update_status()

    def on_radius_change(self) -> None:
        if self._suppress_radius or self.selected is None:
            return
        if self._warn_locked("resize points"):
            return
        side, idx = self.selected
        panel = self._panel(side)
        if idx >= len(panel.points):
            self._select(None, None)
            return
        r = max(RADIUS_MIN, min(RADIUS_MAX, int(round(self.radius.get()))))
        pt = panel.points[idx]
        if pt.r != r:
            pt.r = r
            self._invalidate_alignment()
            panel.draw()
            self._update_status()

    # ----------------------------------------------------------------- lists

    def clear_points(self) -> None:
        if self._warn_locked("clear points"):
            return
        self.panel_top.points.clear()
        self.panel_bottom.points.clear()
        self._select(None, None)
        self._invalidate_alignment()
        self.panel_top.draw(); self.panel_bottom.draw()
        self._update_status()

    # ------------------------------------------------------------- alignment

    def _is_aligned(self) -> bool:
        return self.transform is not None

    def _warn_locked(self, action: str) -> bool:
        if not self._is_aligned():
            return False
        self.status.config(
            text=f"Alignment is locked — click 'Reset Alignment' first to {action}.")
        return True

    def _invalidate_alignment(self) -> None:
        self.transform = None
        self.warped_bottom = None
        for v in self._pad_views():
            v.set_pair(None, None)
        self._refresh_mode_bars()

    def _on_align_click(self) -> None:
        if getattr(self.align_btn, "_enabled", True):
            self.align()

    def align(self) -> None:
        if self._is_aligned():
            return
        n = min(len(self.panel_top.points), len(self.panel_bottom.points))
        if n < 3:
            messagebox.showinfo("Not enough points", "Add at least 3 matching point pairs.")
            return
        if self.panel_top.image is None or self.panel_bottom.image is None:
            return
        src = [(p.x, p.y) for p in self.panel_bottom.points[:n]]
        dst = [(p.x, p.y) for p in self.panel_top.points[:n]]
        M, kind = fit_transform(src, dst)
        residual = transform_residual(src, dst, M)
        self.transform = M
        self.warped_bottom = warp_image(self.panel_bottom.image, M, self.panel_top.image.size)
        for v in self._pad_views():
            v.set_pair(self.panel_top.image, self.warped_bottom)
            v.set_alpha(self.opacity.get())
        self.set_mode("overlay")
        self._refresh_mode_bars()
        self.status.config(
            text=f"Aligned {n} points · {kind} · mean residual {residual:.2f}px"
                 f"  ·  alignment locked")

    def reset_alignment(self) -> None:
        if not self._is_aligned():
            return
        if self.overlay.pads:
            if not self._confirm_reset_with_pads():
                return
        elif not messagebox.askyesno(
            "Reset alignment?",
            "Clear the current alignment? You can re-align afterwards."):
            return
        self.transform = None
        self.warped_bottom = None
        self.single_image_mode = False
        for v in self._pad_views():
            v.set_pair(None, None)
        self._refresh_mode_bars()
        self._update_status()
        self.status.config(text="Alignment reset — adjust points and click Align.")

    def _confirm_reset_with_pads(self) -> bool:
        win = tk.Toplevel(self.root)
        win.title("Reset alignment?")
        win.transient(self.root)
        win.resizable(False, False)
        try:
            win.grab_set()
        except tk.TclError:
            pass

        frm = ttk.Frame(win, padding=16)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Reset the alignment?",
                  font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        n = len(self.overlay.pads)
        ttk.Label(
            frm, justify="left", padding=(0, 8),
            text=(f"You have {n} named pad{'s' if n != 1 else ''}.\n\n"
                  "Pad positions were placed against the CURRENT alignment.\n"
                  "After re-aligning, they may no longer line up with the\n"
                  "actual board features and you may need to reposition them.\n\n"
                  "Pads themselves will not be deleted."),
        ).pack(anchor="w")

        confirmed = {"v": False}
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")

        def do_reset():
            confirmed["v"] = True
            win.destroy()

        ttk.Button(btns, text="I understand the risk — reset",
                   command=do_reset).pack(side="right", padx=(0, 6))
        win.bind("<Escape>", lambda e: win.destroy())
        win.wait_window()
        return confirmed["v"]

    # ----------------------------------------------------------------- modes

    def set_mode(self, mode: str) -> None:
        self.mode.set(mode)
        for w in (self.panel_top, self.panel_bottom, self.overlay):
            w.pack_forget()
        self.sbs_frame.pack_forget()
        self.overlay_left.canvas.grid_forget()
        self.overlay_right.canvas.grid_forget()
        if not self._is_aligned():
            self.panel_top.pack(side="left", padx=4, fill="both", expand=True)
            self.panel_bottom.pack(side="left", padx=4, fill="both", expand=True)
            self.panel_top.draw(); self.panel_bottom.draw()
        elif mode == "overlay":
            self.overlay.pack(fill="both", expand=True, padx=4)
        else:
            self.sbs_frame.pack(fill="both", expand=True)
            self.overlay_left.canvas.grid(row=0, column=0, padx=4, sticky="nsew")
            self.overlay_right.canvas.grid(row=0, column=1, padx=4, sticky="nsew")
        self._update_status()

    def on_opacity_change(self) -> None:
        for v in self._pad_views():
            v.set_alpha(self.opacity.get())

    def rotate_overlay(self) -> None:
        if self.warped_bottom is None:
            messagebox.showinfo("Nothing to rotate", "Press Align first.")
            return
        for v in self._pad_views():
            v.rotate_cw()

    def flip_overlay(self) -> None:
        if self.warped_bottom is None:
            messagebox.showinfo("Nothing to flip", "Press Align first.")
            return
        for v in self._pad_views():
            v.toggle_flip()

    def fit_views(self) -> None:
        if not self._is_aligned():
            self.panel_top.fit(); self.panel_bottom.fit()
        elif self.mode.get() == "overlay":
            self.overlay.fit()
        else:
            self.overlay_left.fit(); self.overlay_right.fit()

    # ---------------------------------------------------------------- status

    def _update_status(self) -> None:
        if self.panel_top.image is None or self.panel_bottom.image is None:
            self.status.config(text="Load a top and bottom image to begin.")
            self._refresh_align_button_style()
            return
        nt, nb = len(self.panel_top.points), len(self.panel_bottom.points)
        msg = f"Top: {nt}  |  Bottom: {nb}"
        if self._is_aligned():
            msg += "  — alignment locked. Click 'Reset Alignment' to redo."
        elif nt != nb:
            msg += "  — pair the next point on the other side"
        elif nt < 3:
            msg += f"  — need {3 - nt} more pair(s) to align"
        else:
            msg += "  — ready to Align" + (" (more points → better fit)" if nt < 8 else "")
        if self.selected is not None:
            side, idx = self.selected
            panel = self._panel(side)
            if idx < len(panel.points):
                pt = panel.points[idx]
                msg += f"   ·  selected #{idx + 1} ({side.upper()})  r={pt.r}px"
        self.status.config(text=msg)
        self._refresh_align_button_style()


def main() -> None:
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
