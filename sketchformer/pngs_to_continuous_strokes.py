#!/usr/bin/env python3
# pngs_to_continuous_strokes.py
"""
Convert PNG sketch images into continuous stroke sequences expected by SketchFormer.
Each stroke vector has 5 values: [dx, dy, pen_down, pen_up, pen_end/pad]
We pad sequences up to max_seq_len with [0,0,0,0,1] (pad token used in the repo).
"""

import os
import numpy as np
import cv2
from skimage.morphology import skeletonize
from glob import glob
import argparse

def preprocess_image(path, target_size=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert so stroke pixels = 1
    if np.mean(th) > 127:
        th = 255 - th
    sk = skeletonize((th > 0))
    sk = (sk.astype(np.uint8) * 255)
    return sk

def image_to_polylines(img, min_length=5, approx_epsilon=2.0):
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

def polylines_to_continuous_seq(polylines, max_seq_len=200):
    """
    Convert list of polylines -> sequence of [dx, dy, p_down, p_up, p_end]
    - interior points: p_down = 1 -> [1,0,0]
    - stroke break: p_up = 1 -> [0,1,0] (zero delta)
    - final padding: [0,0,0,0,1]
    """
    seq = []
    for i, poly in enumerate(polylines):
        if poly.shape[0] < 2:
            continue
        # convert poly points to sequence of deltas
        deltas = np.diff(poly, axis=0)  # shape (n-1,2)
        for j in range(deltas.shape[0]):
            dx, dy = deltas[j]
            # pen_down for movement
            seq.append([float(dx), float(dy), 1.0, 0.0, 0.0])
        # between strokes -> pen_up marker (zero delta)
        if i < len(polylines) - 1:
            seq.append([0.0, 0.0, 0.0, 1.0, 0.0])

    if not seq:
        # fallback: single zero move + pen_end
        seq = [[0.0, 0.0, 0.0, 0.0, 1.0]]

    # truncate if too long
    if len(seq) > max_seq_len:
        print(len(seq))
        seq = seq[:max_seq_len]
    # pad to max_seq_len with [0,0,0,0,1]
    pad_token = [0.0, 0.0, 0.0, 0.0, 1.0]
    while len(seq) < max_seq_len:
        seq.append(pad_token)
    return np.array(seq, dtype=np.float32)

def img_path_to_seq(path, target_size=None, max_seq_len=200):
    sk = preprocess_image(path, target_size=target_size)
    polylines = image_to_polylines(sk, min_length=8, approx_epsilon=2.0)
    seq = polylines_to_continuous_seq(polylines, max_seq_len=max_seq_len)
    return seq

def main(png_folder, out_npz, target_size=None, max_seq_len=200):
    paths = sorted(glob(os.path.join(png_folder, '*.png')))
    X = []
    filenames = []
    for p in paths:
        try:
            seq = img_path_to_seq(p, target_size=target_size, max_seq_len=max_seq_len)
            X.append(seq)  # each seq shape (max_seq_len, 5)
            filenames.append(os.path.basename(p))
            print("Converted:", os.path.basename(p), "len:", np.sum(seq[..., -1] != 1))
        except Exception as e:
            print("Failed:", p, e)
    if not X:
        raise RuntimeError("No sequences produced.")
    X = np.stack(X, axis=0)  # shape (N, max_seq_len, 5)
    np.savez(out_npz, x=X, filenames=np.array(filenames))
    print("Saved dataset:", out_npz, "shape:", X.shape)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--png-folder", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--max-seq-len", type=int, default=200)
    p.add_argument("--target-size", type=int, nargs=2, default=None)
    args = p.parse_args()
    main(args.png_folder, args.out_npz, target_size=tuple(args.target_size) if args.target_size else None,
         max_seq_len=args.max_seq_len)

"""
Execution code
pip install scikit-image opencv-python-headless
python pngs_to_continuous_strokes.py --png-folder ./input_png --out-npz ./sketchformer_dataset/chunk_0.npz --max-seq-len 200

"""