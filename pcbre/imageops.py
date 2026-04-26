"""Affine + homography fitting and perspective image warping."""

from __future__ import annotations

import numpy as np
from PIL import Image

# PCB photos are routinely tens of megapixels.
Image.MAX_IMAGE_PIXELS = None

try:
    PERSPECTIVE = Image.Transform.PERSPECTIVE
    BILINEAR = Image.Resampling.BILINEAR
    ROTATE_90 = Image.Transpose.ROTATE_90
    ROTATE_180 = Image.Transpose.ROTATE_180
    ROTATE_270 = Image.Transpose.ROTATE_270
except AttributeError:  # older Pillow
    PERSPECTIVE = Image.PERSPECTIVE
    BILINEAR = Image.BILINEAR
    ROTATE_90 = Image.ROTATE_90
    ROTATE_180 = Image.ROTATE_180
    ROTATE_270 = Image.ROTATE_270


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: center on mean, scale so mean dist = sqrt(2)."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    d = float(np.linalg.norm(pts - c, axis=1).mean())
    if d < 1e-9:
        return pts.copy(), np.eye(3)
    s = np.sqrt(2.0) / d
    T = np.array([[s, 0.0, -s * c[0]],
                  [0.0, s, -s * c[1]],
                  [0.0, 0.0, 1.0]])
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    return (T @ pts_h.T).T[:, :2], T


def compute_affine(src_pts, dst_pts) -> np.ndarray:
    """3x3 affine (last row [0,0,1]) from N>=3 correspondences."""
    src_n, T_s = _normalize(src_pts)
    dst_n, T_d = _normalize(dst_pts)
    A, b = [], []
    for (x, y), (xp, yp) in zip(src_n, dst_n):
        A.append([x, y, 1, 0, 0, 0]); b.append(xp)
        A.append([0, 0, 0, x, y, 1]); b.append(yp)
    sol, *_ = np.linalg.lstsq(np.asarray(A, float),
                              np.asarray(b, float), rcond=None)
    M_n = np.array([[sol[0], sol[1], sol[2]],
                    [sol[3], sol[4], sol[5]],
                    [0.0,    0.0,    1.0   ]])
    return np.linalg.inv(T_d) @ M_n @ T_s


def compute_homography(src_pts, dst_pts) -> np.ndarray:
    """3x3 homography from N>=4 correspondences via normalized DLT."""
    src_n, T_s = _normalize(src_pts)
    dst_n, T_d = _normalize(dst_pts)
    A = []
    for (x, y), (xp, yp) in zip(src_n, dst_n):
        A.append([-x, -y, -1,  0,  0,  0, x * xp, y * xp, xp])
        A.append([ 0,  0,  0, -x, -y, -1, x * yp, y * yp, yp])
    _, _, Vt = np.linalg.svd(np.asarray(A, float))
    H = np.linalg.inv(T_d) @ Vt[-1].reshape(3, 3) @ T_s
    return H / H[2, 2]


def fit_transform(src_pts, dst_pts) -> tuple[np.ndarray, str]:
    """Affine for 3 pairs, homography for >=4."""
    n = min(len(src_pts), len(dst_pts))
    if n < 3:
        raise ValueError("need at least 3 point pairs")
    if n >= 4:
        return compute_homography(src_pts[:n], dst_pts[:n]), "homography"
    return compute_affine(src_pts[:n], dst_pts[:n]), "affine"


def transform_residual(src_pts, dst_pts, M: np.ndarray) -> float:
    """Mean reprojection error of M·src vs dst, in pixels."""
    s = np.column_stack([np.asarray(src_pts, float), np.ones(len(src_pts))])
    proj = (M @ s.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return float(np.linalg.norm(proj - np.asarray(dst_pts, float), axis=1).mean())


def warp_image(src: Image.Image, M: np.ndarray, out_size) -> Image.Image:
    """Warp `src` to `out_size` using 3x3 matrix `M` mapping src→dst."""
    Minv = np.linalg.inv(M)
    Minv = Minv / Minv[2, 2]
    coeffs = (Minv[0, 0], Minv[0, 1], Minv[0, 2],
              Minv[1, 0], Minv[1, 1], Minv[1, 2],
              Minv[2, 0], Minv[2, 1])
    return src.transform(out_size, PERSPECTIVE, coeffs, resample=BILINEAR)
