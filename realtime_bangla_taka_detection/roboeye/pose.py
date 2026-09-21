"""3D note pose from a detected box using OpenCV solvePnP.

Finds a rotated rectangle (note contour) inside the YOLO crop, then recovers
roll / pitch / yaw relative to the camera. Large tilt triggers a
“straighten the note” prompt.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import NOTE_SIZE_MM

TILT_LIMIT_DEG = 28.0


def _camera_matrix(w: int, h: int) -> np.ndarray:
    f = float(max(w, h))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)


def _object_corners(name: str) -> np.ndarray:
    L, W = NOTE_SIZE_MM.get(name, (140.0, 62.0))
    hw, hh = L / 2.0, W / 2.0
    return np.array(
        [[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]],
        dtype=np.float64,
    )


def _ordered_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as tl, tr, br, bl."""
    pts = np.array(pts, dtype=np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def _rotated_rect_in_crop(crop_bgr: np.ndarray) -> np.ndarray | None:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 80:
        h, w = gray.shape
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return box.astype(np.float64)


def rpy_from_rvec(rvec: np.ndarray) -> tuple[float, float, float]:
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def estimate_pose(
    frame_bgr: np.ndarray,
    xyxy: tuple[int, int, int, int],
    class_name: str,
) -> dict:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    crop = frame_bgr[y1:y2, x1:x2]
    quad = _rotated_rect_in_crop(crop)
    if quad is None:
        return {"ok": False, "needs_straighten": False}
    quad[:, 0] += x1
    quad[:, 1] += y1
    image_pts = _ordered_quad(quad)
    obj = _object_corners(class_name)
    K = _camera_matrix(w, h)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(obj, image_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {"ok": False, "needs_straighten": False, "quad": image_pts}
    roll, pitch, yaw = rpy_from_rvec(rvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    reproj = float(np.linalg.norm(proj.reshape(-1, 2) - image_pts, axis=1).mean())
    tilt = max(abs(roll), abs(pitch))
    return {
        "ok": True,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "tvec": tvec.reshape(3),
        "rvec": rvec.reshape(3),
        "quad": image_pts,
        "reproj_error_px": reproj,
        "needs_straighten": tilt > TILT_LIMIT_DEG,
        "tilt": tilt,
    }


def evaluate_pnp(
    n_samples: int = 400,
    noise_px: float = 3.05,
    seed: int = 7,
    image_size: tuple[int, int] = (640, 480),
    class_name: str = "100_taka",
) -> dict:
    """Synthetic solvePnP study: project known note corners, add pixel noise, recover pose."""
    rng = np.random.default_rng(seed)
    w, h = image_size
    K = _camera_matrix(w, h)
    dist = np.zeros((4, 1))
    obj = _object_corners(class_name)
    reproj, rolls, pitches, yaws = [], [], [], []
    for _ in range(n_samples):
        rvec_gt = np.deg2rad(
            np.array(
                [
                    rng.normal(0.0, 3.25),
                    rng.normal(0.0, 4.20),
                    rng.normal(0.0, 3.60),
                ],
                dtype=np.float64,
            )
        ).reshape(3, 1)
        tvec_gt = np.array([[0.0], [0.0], [280.0 + rng.normal(0, 20)]], dtype=np.float64)
        proj, _ = cv2.projectPoints(obj, rvec_gt, tvec_gt, K, dist)
        image_pts = proj.reshape(4, 2) + rng.normal(0.0, noise_px, size=(4, 2))
        ok, rvec, tvec = cv2.solvePnP(obj, image_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        pred, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        reproj.append(float(np.linalg.norm(pred.reshape(-1, 2) - image_pts, axis=1).mean()))
        roll, pitch, yaw = rpy_from_rvec(rvec)
        rolls.append(abs(roll))
        pitches.append(abs(pitch))
        yaws.append(abs(yaw))
    return {
        "n_samples": len(reproj),
        "mean_reproj_error_px": float(np.mean(reproj)) if reproj else None,
        "mean_abs_roll_deg": float(np.mean(rolls)) if rolls else None,
        "mean_abs_pitch_deg": float(np.mean(pitches)) if pitches else None,
        "mean_abs_yaw_deg": float(np.mean(yaws)) if yaws else None,
        "noise_px": noise_px,
    }
