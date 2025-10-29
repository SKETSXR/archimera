import os
import warnings

# Suppress TensorFlow and Keras logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # prevents verbose GPU memory messages
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"          # suppress absl GPU device INFO
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import faiss
import numpy as np
import json
import torch
from PIL import Image
from pngs_to_continuous_strokes import img_path_to_seq
from extract_sketchformer_embeddings import build_model_and_restore

def extract_embedding_from_npz(npz: np.ndarray, model):
    try:
        out = model.encode_from_seq(inp_seq=npz)
        # print(type(out))
        emb = out['embedding']
        if hasattr(emb, 'numpy'):
            emb_np = emb.numpy()
        else:
            emb_np = np.array(emb)
        if emb_np.ndim == 3:
            emb_np = emb_np.mean(axis=1)
        emb_np = emb_np.reshape(-1)
        # print(f"Embedding Generated: {emb_np.shape}")
        return emb_np.astype('float32')
    except Exception as e:
        print(f"ERROR Embedding:- {e}")


if __name__ == "__main__":
    
    weights_dir = "./sketch-transformer-tf2-cvpr_tform_cont/weights"
    config_json = "./sketch-transformer-tf2-cvpr_tform_cont/config.json"
    faiss_index = "./sketchformer_index.faiss"
    mapping_path = "./id_mapping_sketchformer.json"
    top_k = 5
    
    # Building Model using weights
    model = build_model_and_restore(weights_dir=weights_dir, max_seq_len=200, config_json=config_json)
    # print("Loaded lowerdim:", model.hps['lowerdim'])
    
    for i in range(1, 9):
        query_path = f"input_png/pdf{i}.png"
        # Converting png to continuous stroke form
        seq = img_path_to_seq(path=query_path, target_size=None, max_seq_len=200)
        # print(f"Sequence Shape before reshaping: {seq.shape}")
        # Adding batch dimension to make it compatible with SKetchFormer
        seq = np.expand_dims(seq, axis=0)
        # print(f"Sequence Shape after reshape: {seq.shape}")
        

        # Extracting Embeddings
        embedding = extract_embedding_from_npz(npz=seq, model=model)
        # Normalizing Embedding for similarity matching
        norms = np.linalg.norm(embedding)
        if norms == 0:
            norms = 1e-10
        emb_norm = embedding / norms
        emb_norm = emb_norm.reshape(1, -1)
        # Loading FAISS index and mapping
        index = faiss.read_index(faiss_index)
        with open(mapping_path, "r") as f:
            id_mapping = json.load(f)
        # Search top-k similar images
        # print("Query embedding dim:", emb_norm.shape[1])
        # print("FAISS index dim:", index.d)
        D, I = index.search(emb_norm, top_k)

        # Prepare results
        results = []
        for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
            fname = id_mapping.get(str(idx)) or id_mapping.get(idx)
            score = dist
            results.append(
                {
                    "rank": rank,
                    "filename": fname,
                    "score": float(score),
                }
            )
        print(f"\n Top similar sketches for pdf{i}: ")
        for r in results:
            print(f"{r['rank']}. {r['filename']} - score: {round(r['score'] * 100, 2)}%")
    