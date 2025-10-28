"""
This script builds a workflow to extract vector embeddings from images. We use CLIP to extract embeddings, with the weights being vit-base-patch32
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# TODO: Image Path to be added
image_path = "page_1.png"
image = Image.open(image_path).convert("RGB")

# ** Loading Inputs and Models to GPU if available 
inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    img_embed = model.get_image_features(**inputs)

img_embed = img_embed / img_embed.norm(p=2, dim=-1, keepdim=True)
embedding_vec = img_embed[0].cpu().numpy()

print(f"Embedding shape: {embedding_vec.shape}")
print(f"First 10 values: {embedding_vec[:10]}")