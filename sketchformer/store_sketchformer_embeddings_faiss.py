"""
Store SketchFormer embeddings into a FAISS index (cosine similarity).

Steps:
1. Load precomputed SketchFormer embeddings from .npy file.
2. Normalize them for cosine similarity.
3. Create a FAISS IndexFlatIP (Inner Product) index.
4. Add embeddings and save index + mapping for later retrieval.
"""

import os
import json
import numpy as np
import faiss

def build_faiss_index_from_embeddings(
        embeddings_path: str,
        mapping_path: str,
        index_output: str = "sketchformer_index.faiss",
        normalized: bool = True,
):
    """
    Build and save a FAISS index using cosine similarity.

    ---
    Parameters:
        embeddings_path (str): Path to .npy file containing (N, D) embeddings.
        mapping_path (str): Path to JSON or text file mapping IDs to filenames.
        index_output (str): Output FAISS index file path.
        normalized (bool): Whether to normalize embeddings before indexing.
    ---
    Returns:
        (faiss.Index, dict): FAISS index and ID→filename mapping.
    """

    # Load embeddings
    embeddings = np.load(embeddings_path).astype('float32')
    print(f"✅ Loaded embeddings: {embeddings.shape}")

    # Normalize embeddings for cosine similarity
    if normalized:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embeddings = embeddings / norms
        print("✅ Normalized embeddings for cosine similarity")

    # Load mapping
    if mapping_path.endswith(".json"):
        with open(mapping_path, "r") as f:
            id_mapping = json.load(f)
    else:
        with open(mapping_path, "r") as f:
            filenames = [line.strip() for line in f.readlines()]
        id_mapping = {i: filenames[i] for i in range(len(filenames))}

    # Create FAISS index (Inner Product for cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"✅ FAISS index created with {index.ntotal} vectors")

    # Save index and mapping
    faiss.write_index(index, index_output)
    print(f"💾 Saved FAISS index at: {index_output}")

    json_output = os.path.splitext(index_output)[0] + "_mapping.json"
    with open(json_output, "w") as f:
        json.dump(id_mapping, f, indent=2)
    print(f"💾 Saved ID mapping at: {json_output}")

    return index, id_mapping


if __name__ == "__main__":
    embeddings_path = "./sketchformer_embeddings.npy"
    mapping_path = "./id_mapping_sketchformer.json"   # adjust if you have another mapping
    index_output = "./sketchformer_index.faiss"

    index, id_mapping = build_faiss_index_from_embeddings(
        embeddings_path=embeddings_path,
        mapping_path=mapping_path,
        index_output=index_output,
        normalized=True
    )
