#!/usr/bin/env python3
# pngs_to_continuous_strokes.py
"""
Convert PNG sketch images into continuous stroke sequences expected by SketchFormer.
Each stroke vector has 5 values: [dx, dy, pen_down, pen_up, pen_end/pad].
Sequences longer than `max_seq_len` are adaptively reduced using average pooling
(for continuous deltas) and voting-based aggregation (for binary pen states).
"""

import os
import numpy as np
import cv2
from skimage.morphology import skeletonize
from glob import glob
import argparse

# -------------------- Image Preprocessing --------------------
def preprocess_image(path, target_size=None):
    """
    Reads an image, binarizes, inverts (so stroke pixels=1), and skeletonizes.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    # Adaptive threshold using Otsu
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) > 127:  # ensure black strokes on white bg
        th = 255 - th
    sk = skeletonize((th > 0))
    return (sk.astype(np.uint8) * 255)


# -------------------- Contour Extraction --------------------
def image_to_polylines(img, min_length=5, approx_epsilon=2.0):
    """
    Extracts skeleton contours and approximates them as polylines.
    """
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polylines = []
    for cnt in contours:
        if len(cnt) < min_length:
            continue
        cnt = cnt.squeeze()
        if cnt.ndim != 2:
            continue
        approx = cv2.approxPolyDP(cnt.astype(np.int32), approx_epsilon, False)
        approx = approx.squeeze()
        if approx.ndim != 2:
            continue
        polylines.append(approx.astype(np.float32))
    return polylines


# -------------------- Polyline → Sequence --------------------
def polylines_to_continuous_seq(polylines):
    """
    Convert polylines to [dx, dy, pen_down, pen_up, pen_end] sequence.
    """
    seq = []
    for i, poly in enumerate(polylines):
        if poly.shape[0] < 2:
            continue
        deltas = np.diff(poly, axis=0)  # shape (n-1,2)
        for dx, dy in deltas:
            seq.append([float(dx), float(dy), 1.0, 0.0, 0.0])  # pen_down
        # mark stroke end
        if i < len(polylines) - 1:
            seq.append([0.0, 0.0, 0.0, 1.0, 0.0])  # pen_up
    if not seq:
        seq = [[0.0, 0.0, 0.0, 0.0, 1.0]]  # fallback single token
    return np.array(seq, dtype=np.float32)


# -------------------- Sequence Pooling --------------------
def pooled_pen_state(chunk, idx):
    """
    Decide binary pen state (pen_down/up/end) for a pooled segment.
    Uses hybrid logic: majority vote with fallback if any '1' present.
    """
    avg = np.mean(chunk[:, idx])
    if avg > 0.5:
        return 1.0
    elif np.any(chunk[:, idx] > 0.5):
        return 1.0
    return 0.0


def average_pool_sequence(seq, target_len):
    """
    Average pool a (N,5) sequence to target_len for SketchFormer compatibility.
    - dx, dy averaged normally
    - pen_* flags aggregated via hybrid voting
    """
    n = len(seq)
    if n <= target_len:
        return seq  # no pooling needed

    step = n / target_len
    pooled = []
    for i in range(target_len):
        start = int(i * step)
        end = int((i + 1) * step)
        chunk = seq[start:end]
        if len(chunk) == 0:
            continue
        dx, dy = np.mean(chunk[:, 0]), np.mean(chunk[:, 1])
        pen_down = pooled_pen_state(chunk, 2)
        pen_up = pooled_pen_state(chunk, 3)
        pen_end = pooled_pen_state(chunk, 4)
        pooled.append([dx, dy, pen_down, pen_up, pen_end])

    return np.array(pooled, dtype=np.float32)


# -------------------- Sequence Padding --------------------
def pad_sequence(seq, max_seq_len):
    """
    Pad sequence to fixed length using [0,0,0,0,1].
    """
    pad_token = [0.0, 0.0, 0.0, 0.0, 1.0]
    if len(seq) > max_seq_len:
        seq = seq[:max_seq_len]
    elif len(seq) < max_seq_len:
        pad = np.tile(pad_token, (max_seq_len - len(seq), 1))
        seq = np.vstack([seq, pad])
    return seq


# -------------------- Conversion Pipeline --------------------
def img_path_to_seq(path, target_size=None, max_seq_len=200):
    """
    Complete conversion from PNG → skeleton → polylines → sequence (pooled + padded).
    """
    sk = preprocess_image(path, target_size=target_size)
    polylines = image_to_polylines(sk, min_length=8, approx_epsilon=2.0)
    seq = polylines_to_continuous_seq(polylines)

    # adaptive pooling if too long
    if len(seq) > max_seq_len:
        seq = average_pool_sequence(seq, max_seq_len)

    seq = pad_sequence(seq, max_seq_len)
    return seq


# -------------------- Main Routine --------------------
def main(png_folder, out_npz, target_size=None, max_seq_len=200):
    paths = sorted(glob(os.path.join(png_folder, '*.png')))
    X, filenames = [], []

    for p in paths:
        try:
            seq = img_path_to_seq(p, target_size=target_size, max_seq_len=max_seq_len)
            X.append(seq)
            filenames.append(os.path.basename(p))
            print(f"✅ Converted {os.path.basename(p)} | Length: {np.sum(seq[..., -1] != 1)}")
        except Exception as e:
            print(f"❌ Failed {p}: {e}")

    if not X:
        raise RuntimeError("No sequences produced.")

    X = np.stack(X, axis=0)  # (N, max_seq_len, 5)
    np.savez(out_npz, x=X, filenames=np.array(filenames))
    print(f"\n💾 Saved dataset: {out_npz}")
    print(f"   Shape: {X.shape}")


# -------------------- Driver Code --------------------
if __name__ == "__main__":
    png_folder = "/home/ayushkum/archimera/inputs/input_png"
    out_npz = "/home/ayushkum/archimera/sketchformer/sketchformer_dataset/chunk_0.npz"
    main(png_folder=png_folder, out_npz=out_npz)

