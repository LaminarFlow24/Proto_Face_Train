import os
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser
import streamlit as st

MODEL_DIR_PATH = 'model'


def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []

    images_per_label = {label: 0 for label in dataset.class_to_idx.values()}

    for img_path, label in dataset.samples:
        st.write(f"Processing {img_path}")
        images_per_label[label] += 1
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            st.write(f"Could not find face on {img_path}")
            continue
        if embedding.shape[0] > 1:
            st.write(f"Multiple faces detected for {img_path}, taking one with highest probability")
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    valid_indices = [i for i, label in enumerate(labels) if images_per_label[label] > 0]

    embeddings = np.array([embeddings[i] for i in valid_indices])
    labels = [labels[i] for i in valid_indices]

    return embeddings, labels


def load_data(dataset_path, embeddings_path, labels_path, class_to_idx_path, features_extractor):
    if embeddings_path:
        return np.loadtxt(embeddings_path), \
               np.loadtxt(labels_path, dtype='str').tolist(), \
               joblib.load(class_to_idx_path)

    dataset = datasets.ImageFolder(dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    return embeddings, labels, dataset.class_to_idx


def train(embeddings, labels, use_grid_search):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    if use_grid_search:
        clf = GridSearchCV(
            estimator=softmax,
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=3
        )
    else:
        clf = softmax
    clf.fit(embeddings, labels)

    return clf.best_estimator_ if use_grid_search else clf


def run_training(dataset_path, embeddings_path, labels_path, class_to_idx_path, use_grid_search):
    features_extractor = FaceFeaturesExtractor()
    embeddings, labels, class_to_idx = load_data(dataset_path, embeddings_path, labels_path, class_to_idx_path, features_extractor)
    clf = train(embeddings, labels, use_grid_search)

    unique_labels = set(labels)
    filtered_class_to_idx = {k: v for k, v in class_to_idx.items() if v in unique_labels}
    idx_to_class = {v: k for k, v in filtered_class_to_idx.items()}

    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    st.write(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))

    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join(MODEL_DIR_PATH, 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)
    st.write(f"Model saved to {model_path}")
