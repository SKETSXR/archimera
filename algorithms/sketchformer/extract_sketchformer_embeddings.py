#!/usr/bin/env python3
# extract_sketchformer_embeddings.py
"""
Load SketchFormer model from sketchformer-master, restore checkpoint, and extract embeddings.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.append("./sketchformer_master")

# make sure repo is on path
from sketchformer_master.models.sketchformer import Transformer as SketchTransformer
from sketchformer_master.utils.hparams import HParams

class DummyDataset:
    """Minimal dataset shim required by model __init__"""
    def __init__(self, max_seq_len=200):
        self.hps = {'use_continuous_data': True, 'max_seq_len': max_seq_len}
        self.n_samples = 1
        self.num_classes = 1
        self.n_classes = 1
        self.tokenizer = None

def load_hparams_from_config(config_path=None):
    """
    Create hparams from default then override with config.json if present.
    """
    hps = SketchTransformer.specific_default_hparams()
    if config_path and os.path.exists(config_path):
        import json
        cfg = json.load(open(config_path))
        for k, v in cfg.items():
            # only update keys that exist; HParams supports item assignment
            try:
                hps[k] = v
            except Exception:
                # ignore unknown keys
                pass
    return hps

def build_model_and_restore(weights_dir, max_seq_len=200, config_json=None):
    dataset = DummyDataset(max_seq_len=max_seq_len)
    hps = load_hparams_from_config(config_json)

    
    # instantiate model
    model = SketchTransformer(hps, dataset, out_dir='.', experiment_id='inference')
    # build model graph: call build_model to create layers
    model.build_model()
    # restore checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(weights_dir)
    if latest is None:
        raise RuntimeError(f"No checkpoint found in {weights_dir}")
    ckpt.restore(latest).expect_partial()
    print("[INFO] Restored checkpoint:", latest)
    return model

def extract_embeddings_from_npz(npz_path, model, out_embeddings_path="sketchformer_embeddings.npy", out_map_path="id_mapping.json"):
    d = np.load(npz_path, allow_pickle=True)
    X = d['x']  # shape (N, seq_len, 5)
    filenames = d.get('filenames', None)
    if filenames is None:
        # try fallback
        filenames = [f"img_{i}.png" for i in range(X.shape[0])]
    embeddings = []
    for i in range(X.shape[0]):
        seq = X[i]  # (seq_len, 5)
        # model.encode_from_seq expects a 1D or 2D array representing the sequence (it casts to np.array)
        seq = np.expand_dims(seq, axis=0)
        try:
            out = model.encode_from_seq(seq)
            emb = out['embedding']  # should be (1, lowerdim) or (1, seq_len, ...)
            # ensure emb is numpy
            if hasattr(emb, 'numpy'):
                emb_np = emb.numpy()
            else:
                emb_np = np.array(emb)
            # if emb has shape (1, d) or (1, seq_len, d), reduce to (d,)
            if emb_np.ndim == 3:
                # average across sequence axis if needed
                emb_np = emb_np.mean(axis=1)
            emb_np = emb_np.reshape(-1)
            embeddings.append(emb_np.astype('float32'))
            print(f"[{i+1}/{X.shape[0]}] Embedded {filenames[i]} -> {emb_np.shape}")
        except Exception as e:
            print("ERROR embedding", filenames[i], e)
            # append zeros to keep indices aligned
            embeddings.append(np.zeros((model.hps['lowerdim'],), dtype='float32'))

    embeddings = np.stack(embeddings, axis=0)
    np.save(out_embeddings_path, embeddings)
    # save mapping
    id_map = {i: str(filenames[i]) for i in range(len(filenames))}
    with open(out_map_path, "w") as f:
        json.dump(id_map, f, indent=2)
    print("[INFO] Saved embeddings:", out_embeddings_path, "shape:", embeddings.shape)
    print("[INFO] Saved mapping:", out_map_path)
    return embeddings, id_map

if __name__ == "__main__":
    npz_path = "./sketchformer_dataset/chunk_0.npz"
    weights_dir = "./sketch-transformer-tf2-cvpr_tform_cont/weights"
    config_json = "./sketch-transformer-tf2-cvpr_tform_cont/config.json"  # optional
    model = build_model_and_restore(weights_dir, max_seq_len=200, config_json=config_json)
    
    extract_embeddings_from_npz(npz_path, model, out_embeddings_path="./sketchformer_embeddings.npy", out_map_path="./id_mapping_sketchformer.json")
