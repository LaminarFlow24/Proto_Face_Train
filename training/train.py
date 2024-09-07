import os
import pickle  # Using pickle instead of joblib
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser

MODEL_DIR_PATH = 'model'


def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []

    # Create a dictionary to track the number of images per label
    images_per_label = {label: 0 for label in dataset.class_to_idx.values()}

    for img_path, label in dataset.samples:
        print(img_path)
        images_per_label[label] += 1
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print(f"Could not find face on {img_path}")
            continue
        if embedding.shape[0] > 1:
            print(f"Multiple faces detected for {img_path}, taking one with the highest probability")
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    # Filter out labels that have 0 images
    valid_indices = [i for i, label in enumerate(labels) if images_per_label[label] > 0]

    embeddings = np.array([embeddings[i] for i in valid_indices])
    labels = [labels[i] for i in valid_indices]

    return embeddings, labels


def load_data(dataset_path=None, embeddings_path=None, labels_path=None, class_to_idx_path=None, features_extractor=None):
    if embeddings_path and labels_path and class_to_idx_path:
        return np.loadtxt(embeddings_path), \
               np.loadtxt(labels_path, dtype='str').tolist(), \
               pickle.load(open(class_to_idx_path, 'rb'))  # Load with pickle

    dataset = datasets.ImageFolder(dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    return embeddings, labels, dataset.class_to_idx


def train(embeddings, labels, grid_search=False):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    if grid_search:
        clf = GridSearchCV(
            estimator=softmax,
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=3
        )
    else:
        clf = softmax
    clf.fit(embeddings, labels)

    return clf.best_estimator_ if grid_search else clf


def run_training(dataset_path, embeddings_path, labels_path, class_to_idx_path, use_grid_search=False):
    features_extractor = FaceFeaturesExtractor()
    embeddings, labels, class_to_idx = load_data(dataset_path, embeddings_path, labels_path, class_to_idx_path, features_extractor)
    clf = train(embeddings, labels, grid_search=use_grid_search)

    # Filter class_to_idx to include only classes that have actual labels
    unique_labels = set(labels)
    filtered_class_to_idx = {k: v for k, v in class_to_idx.items() if v in unique_labels}

    # Create idx_to_class from the filtered class_to_idx
    idx_to_class = {v: k for k, v in filtered_class_to_idx.items()}

    # Generate target names for only the classes that have labels
    target_names = [i[1] for i in sorted(idx_to_class.items(), key=lambda i: i[0])]

    # Print classification report using the filtered target names
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=target_names))

    # Save the model using pickle
    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join(MODEL_DIR_PATH, 'face_recogniser.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_file)

    return model_path
