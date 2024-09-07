import streamlit as st
import zipfile
import os
import subprocess
from pathlib import Path

# Set the Streamlit app title
st.title("Upload and Extract ZIP File")

# Upload ZIP file
uploaded_file = st.file_uploader("Please upload a ZIP file", type="zip")

if uploaded_file is not None:
    # Specify a folder to store the extracted files
    extract_path = Path("extracted_data")
    
    # Create the extraction folder if it doesn't exist
    extract_path.mkdir(exist_ok=True)
    
    # Save the uploaded ZIP file locally
    zip_path = extract_path / "uploaded.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("ZIP file uploaded successfully!")

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    st.success("ZIP file extracted!")

    # Find the folder inside the extracted path
    extracted_folder = None
    for item in extract_path.iterdir():
        if item.is_dir():
            extracted_folder = item
            break

    if extracted_folder:
        extracted_folder_path = str(extracted_folder)
        st.write(f"Extracted folder path: {extracted_folder_path}")

        # Run the Python script using the extracted folder path (handle spaces by quoting the path)
        command = f'python -m training.train -d "{extracted_folder_path}"'

        if st.button("Run Training Script"):
            try:
                # Run the training script
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                st.text(f"Command output:\n{result.stdout}")
                st.text(f"Command errors (if any):\n{result.stderr}")
                
                # Check if the model file exists after training
                model_path = Path("model/face_recogniser.pkl")
                if model_path.exists():
                    st.success("Model trained successfully!")

                    # Provide a download button for the trained model
                    with open(model_path, "rb") as f:
                        st.download_button(
                            label="Download Trained Model",
                            data=f,
                            file_name="face_recogniser.pkl",
                            mime="application/octet-stream"
                        )
                else:
                    st.error("Model file not found after training!")
                    
            except Exception as e:
                st.error(f"An error occurred while running the script: {e}")
    else:
        st.error("No folder found inside the extracted ZIP file!")
