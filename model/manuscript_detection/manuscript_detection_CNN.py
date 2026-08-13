###
# Encode the CNN build to sort a submitted document into one of the manuscripts_non_chiffres
# clusters: M (manuscript), NM (non-manuscript) or U (unrelated)
###
import os
import sys
from pathlib import Path
import argparse
import csv
import json
import random
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
import keras
from keras.layers import *

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.utils.remote_listdir import listdir as remote_listdir

#dataset folder root - absolute path on the data host, so it resolves the
#same regardless of the process cwd
DATASET_PATH = os.path.expanduser('~/Documents/idmc/master1/internship/caramba/')

#M/NM/U corpus: one subfolder per cluster, each holding img/ (the pictures) and metadata/ (one
#<name>.json per picture with a "document_type" field matching the cluster, eg {"document_type": "M"}).
#See manuscripts_non_chiffres/README.md
MANUSCRIPTS_PATH = f"{DATASET_PATH}/manuscripts_non_chiffres"

# Matches the fixed 100x100 canvas used project-wide (see model/preprocessing)
IMG_SIZE = (100, 100)

# Clusters targeted by the classifier, in a fixed order so class index <-> label stays stable
# across training runs (labels loaded from metadata are turned into this index, see
# load_custom_dataset). The one that should pass the gate is MANUSCRIPT_LABEL.
CLUSTER_LABELS = ('M', 'NM', 'U')
MANUSCRIPT_LABEL = 'M'

# NM is under-represented next to M/U (46 vs ~100 samples on the reference corpus), which would
# starve the classifier of NM examples. For each cluster listed here, load_custom_dataset adds one
# extra cropped patch per image (same label) alongside the normal full-image resize, doubling its
# sample count without needing new data.
PATCH_CLUSTERS = ('NM',)
PATCH_SCALE = 0.7  # patch side, as a fraction of the source image's shorter side
PATCH_SEED = 42  # keeps patch placement reproducible across runs, like train_test_split's random_state

# Every document rejected as NM/U (or M below threshold) is kept here for later review
REJECTED_LOG_PATH = "rejected_documents.csv"

# Every document validated as manuscript (M) is copied here for later review
VALIDATED_IMAGES_DIR = f"{DATASET_PATH}/validated_documents"

# Trained model is persisted here so `main()` doesn't need to be retrained on every CLI call
# (see the __main__ block at the bottom of this file)
MODEL_PATH = f"{DATASET_PATH}/models/manuscript_detection_cnn.keras"

###
#BODY
###


def extract_patch(img, img_size, scale, rng):
    """Crop a random square patch covering `scale` of img's shorter side and resize it to
    img_size - a lightweight augmentation used to pad out an under-represented cluster's
    sample count without depending on new data. img is the full-resolution PIL image, not
    the already-downscaled training sample."""
    width, height = img.size
    patch_side = max(1, int(min(width, height) * scale))
    max_left = width - patch_side
    max_top = height - patch_side
    left = rng.randint(0, max_left) if max_left > 0 else 0
    top = rng.randint(0, max_top) if max_top > 0 else 0
    patch = img.crop((left, top, left + patch_side, top + patch_side))
    return patch.resize(img_size)


def load_custom_dataset(dataset_root, clusters=CLUSTER_LABELS, img_size=IMG_SIZE,
                         patch_clusters=PATCH_CLUSTERS, patch_scale=PATCH_SCALE):
    images = []
    labels = []
    rng = random.Random(PATCH_SEED)

    for cluster in clusters:
        images_dir = f"{dataset_root}/{cluster}/img"
        labels_dir = f"{dataset_root}/{cluster}/metadata"

        # Sort so images and labels line up consistently
        image_files = sorted(remote_listdir(images_dir))

        for fname in image_files:
            name, ext = os.path.splitext(fname)
            img_path = os.path.join(images_dir, fname)
            label_path = os.path.join(labels_dir, name + ".json")

            if not os.path.exists(label_path):
                print(f"Warning: no label found for {fname}, skipping.")
                continue

            # Load label - metadata's document_type is one of CLUSTER_LABELS, turned into
            # its index so it can be used as a sparse-categorical target below
            with open(label_path, "r") as f:
                label_data = json.load(f)
            label_index = clusters.index(label_data["document_type"])

            # Load image, kept at full resolution until after the patch is cropped from it
            full_img = Image.open(img_path).convert("RGB")

            images.append(np.array(full_img.resize(img_size)))
            labels.append(label_index)

            # Add a cropped patch of the same image under the same label, doubling this
            # cluster's sample count (see PATCH_CLUSTERS)
            if cluster in patch_clusters:
                images.append(np.array(extract_patch(full_img, img_size, patch_scale, rng)))
                labels.append(label_index)

    x = np.array(images, dtype=np.uint8)
    y = np.array(labels, dtype=np.int64).reshape(-1, 1)  # match cifar10's shape (N,1)

    return x, y


#Build layers => change size and activation function
def build_model(img_size=IMG_SIZE, num_classes=len(CLUSTER_LABELS)):
    model = models.Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(*img_size, 3)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #Output layer: one softmax unit per cluster (M / NM / U), since a document belongs to exactly one
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    #sparse_categorical_crossentropy since labels are class indices (N,1), not one-hot vectors
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model


#Load the M/NM/U corpus, train the CNN on it and report held-out accuracy. Returns the trained
#model so callers (eg predict_manuscript) can reuse it without retraining.
def main(dataset_root=MANUSCRIPTS_PATH, epochs=10):
    x_all, y_all = load_custom_dataset(dataset_root)

    train_images, test_images, train_labels, test_labels = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = build_model()
    model.summary()

    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)

    # Persisted so a later CLI call can classify documents without retraining (see __main__ below)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    return model


###
#Gating: use a trained model to accept/reject a single submitted document
###


def log_rejected_document(image_path, predicted_cluster, probability):
    file_exists = os.path.exists(REJECTED_LOG_PATH)
    with open(REJECTED_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_path", "predicted_cluster", "probability"])
        writer.writerow([image_path, predicted_cluster, probability])


def save_validated_document(image_path):
    os.makedirs(VALIDATED_IMAGES_DIR, exist_ok=True)
    img = cv2.imread(image_path)
    cv2.imwrite(os.path.join(VALIDATED_IMAGES_DIR, os.path.basename(image_path)), img)


def predict_manuscript(model, image_path, threshold=0.5):
    """Gate a user-submitted document: classify it into M/NM/U with the given trained model and
    reject it unless it's confidently a manuscript (M), before it reaches preprocessing,
    clustering or classification."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.uint8)[np.newaxis, ...] / 255.0

    probabilities = model.predict(x, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_cluster = CLUSTER_LABELS[predicted_index]
    probability = float(probabilities[predicted_index])

    is_manuscript = predicted_cluster == MANUSCRIPT_LABEL and probability >= threshold

    if not is_manuscript:
        log_rejected_document(image_path, predicted_cluster, probability)
    else:
        save_validated_document(image_path)

    return is_manuscript


#Load the model saved by main() at MODEL_PATH, training one from scratch (which also saves it)
#if none exists yet or --retrain was passed
def _load_or_train_model(retrain=False):
    if not retrain and os.path.exists(MODEL_PATH):
        print(f"Loading trained model from {MODEL_PATH}")
        return models.load_model(MODEL_PATH)
    return main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="*",
                         help="Path(s) to image(s) to classify as M/NM/U. Omit to just "
                              "(re)train the model and exit.")
    parser.add_argument("--retrain", action="store_true",
                         help="Retrain from MANUSCRIPTS_PATH even if a saved model already "
                              "exists at MODEL_PATH.")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Minimum M probability to accept a document as a manuscript "
                              "(default 0.5).")
    args = parser.parse_args()

    #No image given: this call is just meant to (re)train, so always go through main()
    trained_model = _load_or_train_model(retrain=args.retrain or not args.image)

    for path in args.image:
        accepted = predict_manuscript(trained_model, path, threshold=args.threshold)
        print(f"{path}: {'ACCEPTED (manuscript)' if accepted else 'REJECTED'}")
