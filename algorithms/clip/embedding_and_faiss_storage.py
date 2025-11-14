"""
This script implements the following:-
1. Encode images into embeddings.
2. Store these embeddings into FAISS.
3. Store a mapping between images and vectors for later retrieval 
"""

import os
import json
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def build_faiss_index(
        image_folder: str,
        index_path: str = "./sketch_index.faiss",
        mapping_path: str = "./id_mapping.json",
        model_name: str = "openai/clip-vit-base-patch32",
        distance_metric: str = "L2",
):
    """
    Build a FAISS index from a folder of images using CLIP embeddings.

    ---
    Parameters:
        image_folder (str): Path to the folder containing images.
        index_path (str): Path where FAISS index file will be saved.
        mapping_path (str): Path to save filename-to-index mapping JSON.
        model_name (str): Pretrained CLIP model to use.
        distance_metric (str): Distance metric to be used for computing similarity. Currently supports "L2" for Euclidean Distance, and "cosine" for cosine similarity.
    
    ---
    Returns:
        tuple (faiss.Index, dict): the faiss Index and filename mapping
    """
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    embeddings = []
    filenames = []

    # Process each image
    for fname in os.listdir(image_folder):
        if fname.lower().endswith(('.png','.jpg', '.jpeg')):
            path = os.path.join(image_folder, fname)
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Skipping {fname}: {e}")
                continue

            inputs = processor(images=image, return_tensors='pt').to(device)
            with torch.no_grad():
                image_embeds = model.get_image_features(**inputs)
            
            # Normalize for cosine or L2 comparison
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            emb = image_embeds[0].cpu().numpy().astype('float32')

            embeddings.append(emb)
            filenames.append(fname)
    
    if not embeddings:
        raise ValueError(f"No valid images found in {image_folder}")
    
    embeddings = np.stack(embeddings, axis=0)
    print(f"Total embeddings computed: {embeddings.shape}")

    # Select distance metric
    dim = embeddings.shape[1]
    if distance_metric.lower() == "cosine":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    
    index.add(embeddings)

    # Save index and mapping
    faiss.write_index(index, index_path)
    print(f"FAISS index saved at: {index_path}")

    id_mapping = {i: filenames[i] for i in range(len(filenames))}
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f, indent=2)
    print(f"Mapping saved at: {mapping_path}")

    return index, id_mapping

# Driver code
if __name__ == '__main__':
    image_folder = "./input_png"
    index_path = "./sketch_index.faiss"
    mapping_path = "./id_mapping.json"
    model_name = "openai/clip-vit-base-patch32"
    distance_metric = "cosine"

    index, mapping = build_faiss_index(
        image_folder=image_folder,
        index_path=index_path,
        mapping_path=mapping_path,
        model_name=model_name,
        distance_metric=distance_metric
    )