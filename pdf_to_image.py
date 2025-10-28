"""
This script converts a pdf file to a png image file. In case of multipage pdf, each page is rendered as a single image
"""

import os
from pdf2image import convert_from_path

def convert_pdfs_to_pngs(input_folder, output_folder, dpi=600):
    """
    Converts all single-page PDFs in a folder to PNGs with matching filenames.

    ---
    Parameters:
        input_folder (str): Path to the folder containing pdf files.
        output_folder (str): Path to the folder where PNGs will be saved.
        dpi (int): Resolution for conversion. The default value is 600
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all PDF files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(output_folder, f"{base_name}.png")

            # Convert PDF to image
            images = convert_from_path(pdf_path, dpi=dpi)
            img = images[0]

            # Save as PNG
            img.save(output_path, "PNG")
            print(f"Converted: {file_name} -> {base_name}.png")
    
    print("\n Conversion complete!")

if __name__ == "__main__":
    input_folder = "input_pdf"
    output_folder = "input_png"
    dpi = 600
    convert_pdfs_to_pngs(input_folder=input_folder, output_folder=output_folder, dpi=dpi)