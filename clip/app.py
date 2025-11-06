"""
We are developing a Streamlit web application that will utilize the CLIP model for finding out similar images based on user input.
The expected functionalities include:
1. Uploading an image or PDF file through the web interface.
2. Processing the uploaded file to extract images (if PDF).
3. Using the CLIP model to generate embeddings for the uploaded image.
4. Comparing the generated embeddings with the stored embeddings in FAISS index to find similar images.
5. Return a table of similar images filenames along with their similarity scores.
"""

import streamlit as st
import pandas as pd
from search_query_image_similarity import search_similar_sketches

def main():
    st.title("Image Similarity Search with CLIP")
    st.write("Upload an image or PDF file to find similar images from the database.")

    uploaded_file = st.file_uploader("Choose an image or PDF file", type=["png", "pdf"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.write("PDF file uploaded. Extracting images is currently not implemented yet.")
        else:
            with open("temp_image.png", "wb") as f:
                f.write(uploaded_file.getbuffer())
                st.image("temp_image.png", caption='Uploaded Image', use_container_width=True, width='stretch')
            
            if st.button("Find Similar Images"):
                results = search_similar_sketches(
                    query_path="temp_image.png",
                    index_path="./sketch_index.faiss",
                    mapping_path="./id_mapping.json",
                    model_name="openai/clip-vit-base-patch32",
                    top_k=5,
                    distance_metric="cosine"
                )

                # Convert to DataFrame
                df = pd.DataFrame(results)
                df = df[["rank", "filename", "score"]]

                # Convert score to percentage
                df["score"] = (df["score"] * 100).round(2)
                df.rename(columns={"score": "score (%)"}, inplace=True)

                st.subheader("Similar Images")
                st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()