import os
import argparse
import pickle  # Replacing joblib with pickle
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser

MODEL_DIR_PATH = 'model'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for training Face Recognition model. You can either give path to dataset or provide path '
                    'to pre-generated embeddings, labels and class_to_idx. You can pre-generate this with '
                    'util/generate_embeddings.py script.')
    parser.add_argument('-d', '--dataset-path', help='Path to folder with images.')
    parser.add_argument('-e', '--embeddings-path', help='Path to file with embeddings.')
    parser.add_argument('-l', '--labels-path', help='Path to file with labels.')
    parser.add_argument('-c', '--class-to-idx-path', help='Path to pickled class_to_idx dict.')
    parser.add_argument('--grid-search', action='store_true',
                        help='If this option is enabled, grid search will be performed to estimate C parameter of '
                             'Logistic Regression classifier. In order to use this option you have to have at least '
                             '3 examples of every class in your dataset. It is recommended to enable this option.')
    return parser.parse_args()


def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    
    # Create a dictionary to track number of images per label
    images_per_label = {label: 0 for label in dataset.class_to_idx.values()}

    for img_path, label in dataset.samples:
        print(img_path)
        images_per_label[label] += 1
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    # Filter out labels that have 0 images
    valid_indices = [i for i, label in enumerate(labels) if images_per_label[label] > 0]

    # Only keep embeddings and labels that correspond to valid indices
    embeddings = np.array([embeddings[i] for i in valid_indices])
    labels = [labels[i] for i in valid_indices]

    return embeddings, labels


def load_data(args, features_extractor):
    if args.embeddings_path:
        return np.loadtxt(args.embeddings_path), \
               np.loadtxt(args.labels_path, dtype='str').tolist(), \
               pickle.load(open(args.class_to_idx_path, 'rb'))  # Using pickle to load the class_to_idx

    dataset = datasets.ImageFolder(args.dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    return embeddings, labels, dataset.class_to_idx


def train(args, embeddings, labels):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    if args.grid_search:
        clf = GridSearchCV(
            estimator=softmax,
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=3
        )
    else:
        clf = softmax
    clf.fit(embeddings, labels)

    return clf.best_estimator_ if args.grid_search else clf


def main():
    args = parse_args()

    features_extractor = FaceFeaturesExtractor()
    embeddings, labels, class_to_idx = load_data(args, features_extractor)
    clf = train(args, embeddings, labels)

    # Filter class_to_idx to only include classes that have actual labels
    unique_labels = set(labels)
    filtered_class_to_idx = {k: v for k, v in class_to_idx.items() if v in unique_labels}

    # Create idx_to_class from the filtered class_to_idx
    idx_to_class = {v: k for k, v in filtered_class_to_idx.items()}

    # Generate target names for only the classes that have labels
    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    
    # Print classification report using the filtered target names
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))

    # Save the model and class_to_idx using pickle
    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join('model', 'face_recogniser.pkl')
    
    # Save the FaceRecogniser object with pickle
    with open(model_path, 'wb') as model_file:
        pickle.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_file)


if __name__ == '__main__':
    main()
