import streamlit as st
from training.train import run_training

st.title("Face Recognition Training")

# Input fields for dataset and embeddings paths
dataset_path = st.text_input("Enter the dataset path (leave blank if using embeddings):", "")
embeddings_path = st.text_input("Enter the embeddings path:", "")
labels_path = st.text_input("Enter the labels path:", "")
class_to_idx_path = st.text_input("Enter the class_to_idx path:", "")
use_grid_search = st.checkbox("Use grid search for hyperparameter tuning")

# Button to run the training
if st.button("Run Training"):
    if dataset_path or embeddings_path:
        st.write("Running training...")
        run_training(dataset_path, embeddings_path, labels_path, class_to_idx_path, use_grid_search)
        st.write("Training complete!")
    else:
        st.write("Please provide a dataset path or embeddings path.")
