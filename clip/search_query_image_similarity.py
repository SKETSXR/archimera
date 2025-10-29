"""
This script implements the following:-
1. Generate embedding for a user entered image.
2. Load the FAISS index created during embedding generation for stored images.
3. Find similar images based on user selected similarity metric.
4. Return those filenames which are similar along with similarity scores.
"""

import faiss
import numpy as np
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def search_similar_sketches(
        query_path: str,
        index_path: str = "./sketch_index.faiss",
        mapping_path: str = "./id_mapping.json",
        model_name: str = "openai/clip-vit-base-patch32",
        top_k: int = 5,
        distance_metric: str = "L2",
):
    """
    Search for similar sketches using CLIP embeddings and a prebuilt FAISS index.

    ---
    Parameters:
        query_path (str): Path to the query image (sketch).
        index_path (str): Path to FAISS index file.
        mapping_path (str): Path to JSON file containing ID -> filename mapping.
        model_name (str): Pretrained CLIP model to use.
        top_k (int): Number of most similar images to retrieve.
        distance_metric (str): 'L2' or 'cosine' for similarity computation.
    
    ---
    Returns:
        list[dict]: Each item contains:
        {
            "rank": int,
            "filename": str,
            "score": float
        }
    """

    # Load FAISS index and mapping
    index = faiss.read_index(index_path)
    with open(mapping_path, "r") as f:
        id_mapping = json.load(f)
    
    # Load CLIP Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Compute query embedding
    image = Image.open(query_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        query_emb = model.get_image_features(**inputs)
    query_emb = query_emb / query_emb.norm(p=2, dim=-1, keepdim=True)
    query_emb = query_emb.cpu().numpy().astype('float32')

    # Search top-k similar images
    D, I = index.search(query_emb, top_k)

    # Prepare results
    results = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        fname = id_mapping.get(str(idx)) or id_mapping.get(idx)
        # Convert distance to similarity if cosine metric is used
        score = dist if distance_metric.lower() == "cosine" else (1 / (1 + dist))
        results.append(
            {
                "rank": rank,
                "filename": fname,
                "score": float(score),
            }
        )
    
    return results


# ** DRIVER CODE
if __name__ == '__main__':
    for i in range(1, 9):
        query_path = f"./input_png/pdf{i}.png"
        index_path = "./sketch_index.faiss"
        mapping_path = "./id_mapping.json"
        top_k = 5
        distance_metric = "cosine"

        results = search_similar_sketches(
            query_path=query_path,
            index_path=index_path,
            mapping_path=mapping_path,
            top_k=top_k,
            distance_metric=distance_metric
        )

        print(f"\n Top similar sketches for pdf{i}: ")
        for r in results:
            print(f"{r['rank']}. {r['filename']} - score: {round(r['score'] * 100, 2)}%")