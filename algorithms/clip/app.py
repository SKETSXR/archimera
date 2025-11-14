"""
We are developing a Streamlit web application that will utilize the CLIP model for finding out similar images based on user input.
The expected functionalities include:
1. Uploading an image or PDF file through the web interface.
2. Processing the uploaded file to extract images (if PDF).
3. Using the CLIP model to generate embeddings for the uploaded image.
4. Comparing the generated embeddings with the stored embeddings in FAISS index to find similar images.
5. Return a table of similar images filenames along with their similarity scores.
"""
# TODO : Add support for multiple file uploads at once.


import streamlit as st
import time
import pandas as pd
from PIL import Image
from search_query_image_similarity import search_similar_sketches

def convert_pdf_to_images(pdf):
    """This function will convert the PDF file (first page only) uploaded by user through Streamlit to PNG image."""
    from pdf2image import convert_from_bytes
    images = convert_from_bytes(pdf.read(), dpi=600)
    return images[0]

def main():
    st.set_page_config(page_title="CLIP Image Similarity Search", layout="wide")
    st.title("Image Similarity Search with CLIP")
    st.caption("Upload a sketch or PDF and find similar images using CLIP embeddings + FAISS index.")

    st.divider()

    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of Results", min_value=1, max_value=5, value=3)

    uploaded_file = st.file_uploader("Choose an image or PDF file", type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"])

    if uploaded_file:
        file_type = uploaded_file.type
        start_time = time.time()

        if file_type == "application/pdf":
            st.info("PDF uploaded - extracting image...")
            with st.spinner("Converting PDF to image..."):
                image = convert_pdf_to_images(uploaded_file)
                image.save("temp_image.png", "PNG")
            st.image("temp_image.png", caption='Extracted Image from PDF', width='stretch')
        else:
            image = Image.open(uploaded_file).convert("RGB")
            image.save("temp_image.png", "PNG")
            st.image("temp_image.png", caption='Uploaded Image', width='stretch')
        
        if st.button("Find Similar Images", width="content"):
            with st.spinner("Searching for similar images..."):
                try:
                    results = search_similar_sketches(
                        query_path="temp_image.png",
                        index_path="./sketch_index.faiss",
                        mapping_path="./id_mapping.json",
                        model_name="openai/clip-vit-base-patch32",
                        top_k=top_k,
                        distance_metric="cosine"
                    )
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    return
                
                df = pd.DataFrame(results)[["rank", "filename", "score"]]
                df["score (%)"] = (df["score"] * 100).round(2)
                df.drop(columns=["score"], inplace=True)

                elapsed = round(time.time() - start_time, 2)
                st.success(f"Found {len(df)} similar images in {elapsed} seconds.")

                st.subheader("Similarity Scores")
                st.dataframe(df, width="stretch")



if __name__ == "__main__":
    main()