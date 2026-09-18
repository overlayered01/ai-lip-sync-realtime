# SPDX-FileCopyrightText: 2024 Ant Group Co., Ltd. (ditto-talkinghead)
# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from ditto-talkinghead ``core/aux_models/{blaze_face,face_mesh,
# mediapipe_landmark478}.py`` (Apache-2.0), which follow the MediaPipe
# reference calculators. Same math as
# ``avtr1_renderer/components/mediapipe_face.py`` (also Apache-2.0, same
# author), re-hosted here on the numpy ``Engine`` API so this package has no
# dependency on the PolyForm-licensed tree.

"""MediaPipe BlazeFace (short-range) + Face Mesh V2 on the numpy Engine API.

    img_rgb --letterbox 128--> BlazeFace --> (N, 17) detections in image px
           --square ROI x1.5, eye-line rotated--> Face Mesh --> (478, 3)

ONNX graphs: ``blaze_face.onnx`` (input ``input`` (1,128,128,3) in [-1,1];
outputs ``regressors`` (1,896,16), ``classificators`` (1,896,1)) and
``face_mesh.onnx`` (input ``input`` (1,256,256,3) in [0,1]; outputs
``Identity`` (1,1,1,1434) landmarks, ``Identity_1`` presence logit).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lpavatar.runtime.onnx_engine import Engine

BLAZE_INPUT_SIZE = 128
BLAZE_NUM_ANCHORS = 896
BLAZE_NUM_COORDS = 16
BLAZE_NUM_KEYPOINTS = 6
BLAZE_STRIDES = (8, 16, 16, 16)
BLAZE_ANCHORS_PER_LAYER = 2
BLAZE_SCORE_THRESH = 0.5
BLAZE_NMS_IOU = 0.3
BLAZE_PAD_VALUE = 127.5

KP_RIGHT_EYE, KP_LEFT_EYE, KP_NOSE, KP_MOUTH, KP_RIGHT_EAR, KP_LEFT_EAR = range(6)

MESH_INPUT_SIZE = 256
MESH_NUM_POINTS = 478
MESH_ROI_SCALE = 1.5
MESH_PRESENCE_THRESH = 0.5


def generate_blaze_anchors() -> np.ndarray:
    """``(896, 4)`` ``[cx, cy, 1, 1]`` anchor centres (fixed_anchor_size)."""
    anchors: list[list[float]] = []
    layer = 0
    n_layers = len(BLAZE_STRIDES)
    while layer < n_layers:
        stride = BLAZE_STRIDES[layer]
        last_same = layer
        while last_same < n_layers and BLAZE_STRIDES[last_same] == stride:
            last_same += 1
        n_per_cell = BLAZE_ANCHORS_PER_LAYER * (last_same - layer)
        fm = int(np.ceil(BLAZE_INPUT_SIZE / stride))
        for y in range(fm):
            for x in range(fm):
                anchors.extend([[(x + 0.5) / fm, (y + 0.5) / fm, 1.0, 1.0]] * n_per_cell)
        layer = last_same
    out = np.asarray(anchors, dtype=np.float32)
    assert out.shape == (BLAZE_NUM_ANCHORS, 4), out.shape
    return out


_ANCHORS = generate_blaze_anchors()


@dataclass(slots=True, frozen=True)
class Letterbox:
    scale: float  # 128 / max(h, w); content anchored top-left


def blaze_preprocess(img_rgb: np.ndarray) -> tuple[np.ndarray, Letterbox]:
    h, w = img_rgb.shape[:2]
    scale = BLAZE_INPUT_SIZE / float(max(h, w))
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = cv2.resize(img_rgb.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((BLAZE_INPUT_SIZE, BLAZE_INPUT_SIZE, 3), BLAZE_PAD_VALUE, dtype=np.float32)
    canvas[:new_h, :new_w, :] = resized
    blob = (canvas / 127.5 - 1.0)[None, ...]
    return np.ascontiguousarray(blob, dtype=np.float32), Letterbox(scale=scale)


def _decode_boxes(raw: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    out = np.empty_like(raw)
    s = float(BLAZE_INPUT_SIZE)
    x_c = raw[:, 0] / s * anchors[:, 2] + anchors[:, 0]
    y_c = raw[:, 1] / s * anchors[:, 3] + anchors[:, 1]
    w = raw[:, 2] / s * anchors[:, 2]
    h = raw[:, 3] / s * anchors[:, 3]
    out[:, 0] = x_c - w / 2.0
    out[:, 1] = y_c - h / 2.0
    out[:, 2] = x_c + w / 2.0
    out[:, 3] = y_c + h / 2.0
    for k in range(BLAZE_NUM_KEYPOINTS):
        off = 4 + 2 * k
        out[:, off] = raw[:, off] / s * anchors[:, 2] + anchors[:, 0]
        out[:, off + 1] = raw[:, off + 1] / s * anchors[:, 3] + anchors[:, 1]
    return out


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def _weighted_nms(dets: np.ndarray, iou_thresh: float) -> np.ndarray:
    if len(dets) == 0:
        return np.zeros((0, BLAZE_NUM_COORDS + 1), dtype=np.float32)
    remaining = np.argsort(-dets[:, 16])
    out: list[np.ndarray] = []
    while remaining.size > 0:
        top = dets[remaining[0]]
        ious = _iou_one_to_many(top[:4], dets[remaining, :4])
        mask = ious > iou_thresh
        overlapping = remaining[mask]
        remaining = remaining[~mask]
        merged = top.copy()
        if overlapping.size > 1:
            coords = dets[overlapping, :BLAZE_NUM_COORDS]
            scores = dets[overlapping, BLAZE_NUM_COORDS : BLAZE_NUM_COORDS + 1]
            total = float(scores.sum())
            merged[:BLAZE_NUM_COORDS] = (coords * scores).sum(axis=0) / total
            merged[BLAZE_NUM_COORDS] = total / overlapping.size
        out.append(merged)
    return np.stack(out, axis=0).astype(np.float32)


def blaze_postprocess(
    regressors: np.ndarray,
    classificators: np.ndarray,
    letterbox: Letterbox,
    *,
    score_thresh: float = BLAZE_SCORE_THRESH,
    iou_thresh: float = BLAZE_NMS_IOU,
    anchors: np.ndarray = _ANCHORS,
) -> np.ndarray:
    """``(N, 17)`` detections in original pixels, best score first."""
    raw = np.asarray(regressors, dtype=np.float32).reshape(BLAZE_NUM_ANCHORS, BLAZE_NUM_COORDS)
    logits = np.asarray(classificators, dtype=np.float32).reshape(BLAZE_NUM_ANCHORS)
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 100.0)))
    keep = scores >= score_thresh
    if not np.any(keep):
        return np.zeros((0, BLAZE_NUM_COORDS + 1), dtype=np.float32)
    boxes = _decode_boxes(raw[keep], anchors[keep])
    dets = _weighted_nms(np.concatenate([boxes, scores[keep, None]], axis=1), iou_thresh)
    dets[:, :BLAZE_NUM_COORDS] *= BLAZE_INPUT_SIZE / letterbox.scale
    return dets


def detect_faces(img_rgb: np.ndarray, *, det: Engine) -> np.ndarray:
    blob, lb = blaze_preprocess(img_rgb)
    out = det(input=blob)
    return blaze_postprocess(out["regressors"], out["classificators"], lb)


def roi_from_detection(det: np.ndarray, *, scale: float = MESH_ROI_SCALE) -> np.ndarray:
    """``(cx, cy, w, h, rotation)`` square ROI with the eye line horizontal."""
    x1, y1, x2, y2 = det[:4]
    side = max(x2 - x1, y2 - y1) * scale
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = det[4 + 2 * KP_RIGHT_EYE], det[5 + 2 * KP_RIGHT_EYE]
    lx, ly = det[4 + 2 * KP_LEFT_EYE], det[5 + 2 * KP_LEFT_EYE]
    angle = -np.arctan2(ry - ly, lx - rx)
    rotation = angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
    return np.array([cx, cy, side, side, rotation], dtype=np.float64)


def roi_corners(roi: np.ndarray) -> np.ndarray:
    cx, cy, w, h, rot = roi
    hw, hh = w / 2.0, h / 2.0
    pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64)
    c, s = np.cos(rot), np.sin(rot)
    r = np.array([[c, s], [-s, c]], dtype=np.float64)
    return (pts @ r + (cx, cy)).astype(np.float32)


def mesh_preprocess(img_rgb: np.ndarray, roi: np.ndarray) -> np.ndarray:
    src = roi_corners(roi)
    d = float(MESH_INPUT_SIZE)
    dst = np.array([[0, 0], [d, 0], [d, d], [0, d]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(
        img_rgb.astype(np.float32), m, (MESH_INPUT_SIZE, MESH_INPUT_SIZE), flags=cv2.INTER_LINEAR
    )
    return np.ascontiguousarray((crop / 255.0)[None, ...], dtype=np.float32)


def mesh_postprocess(points_flat: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """``(1434,)`` -> ``(478, 3)`` in original image pixels."""
    pts = np.asarray(points_flat, dtype=np.float64).reshape(MESH_NUM_POINTS, 3) / MESH_INPUT_SIZE
    cx, cy, w, h, rot = roi
    x = pts[:, 0] - 0.5
    y = pts[:, 1] - 0.5
    c, s = np.cos(rot), np.sin(rot)
    out = np.empty_like(pts)
    out[:, 0] = (x * c - y * s) * w + cx
    out[:, 1] = (x * s + y * c) * h + cy
    out[:, 2] = pts[:, 2] * w
    return out.astype(np.float32)


def face_mesh_landmarks(
    img_rgb: np.ndarray, roi: np.ndarray, *, mesh: Engine
) -> tuple[np.ndarray, float]:
    out = mesh(input=mesh_preprocess(img_rgb, roi))
    pts = mesh_postprocess(out["Identity"].reshape(-1), roi)
    logit = float(np.asarray(out["Identity_1"]).reshape(-1)[0])
    return pts, float(1.0 / (1.0 + np.exp(-logit)))


def detect_landmarks478(
    img_rgb: np.ndarray,
    *,
    det: Engine,
    mesh: Engine,
    presence_thresh: float = MESH_PRESENCE_THRESH,
    keep_z: bool = False,
) -> np.ndarray:
    """Largest face -> ``(478, 2)`` landmarks in image pixels (``(478, 3)`` with
    ``keep_z``; z is in the same pixel scale as x)."""
    dets = detect_faces(img_rgb, det=det)
    if len(dets) == 0:
        raise RuntimeError("No face detected (BlazeFace).")
    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    roi = roi_from_detection(dets[int(np.argmax(areas))])
    pts, presence = face_mesh_landmarks(img_rgb, roi, mesh=mesh)
    if presence < presence_thresh:
        raise RuntimeError(f"Face Mesh presence {presence:.3f} < {presence_thresh}")
    return np.ascontiguousarray(pts[:, : (3 if keep_z else 2)], dtype=np.float32)


__all__ = [
    "Letterbox",
    "blaze_postprocess",
    "blaze_preprocess",
    "detect_faces",
    "detect_landmarks478",
    "face_mesh_landmarks",
    "generate_blaze_anchors",
    "mesh_postprocess",
    "mesh_preprocess",
    "roi_corners",
    "roi_from_detection",
]
