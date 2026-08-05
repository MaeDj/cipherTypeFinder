###
# Encode the CNN build to distinguish manuscript from random images
###
import os
import sys
from pathlib import Path
import csv
import json
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
DATASET_PATH = os.path.expanduser('~/Caramba/Dataset/corpus_cipherTypeFinder_Caramba')

###
#BODY
###



# Matches the fixed 100x100 canvas used project-wide (see model/preprocessing)
IMG_SIZE = (100, 100)


def load_custom_dataset(images_dir, labels_dir, img_size=IMG_SIZE):
    images = []
    labels = []

    # Sort so images and labels line up consistently
    image_files = sorted(remote_listdir(images_dir))

    for fname in image_files:
        name, ext = os.path.splitext(fname)
        img_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, name + ".json")

        if not os.path.exists(label_path):
            print(f"Warning: no label found for {fname}, skipping.")
            continue

        # Load image
        img = Image.open(img_path).convert("RGB").resize(img_size)
        images.append(np.array(img))

        # Load label
        with open(label_path, "r") as f:
            label_data = json.load(f)
        labels.append(label_data["cipher_types"])

    x = np.array(images, dtype=np.uint8)
    y = np.array(labels, dtype=np.int64).reshape(-1, 1)  # match cifar10's shape (N,1)

    return x, y


# Load everything, then split into train/test
x_all, y_all = load_custom_dataset(f"{DATASET_PATH}/corpus_manuscript_random/images", f"{DATASET_PATH}/corpus_manuscript_random/labels")

train_images, test_images, train_labels, test_labels = train_test_split(
    x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#Build layers => change size and activation function
model = models.Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(*IMG_SIZE, 3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


#To remove
model.summary()

#Output layer
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#Compile
model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

#Fit/train
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

#Model evaluation
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

###
#Main function
###

# Every document rejected as non-manuscript is kept here for later review
REJECTED_LOG_PATH = "rejected_documents.csv"

# Every document validated as manuscript is copied here for later review
VALIDATED_IMAGES_DIR = f"{DATASET_PATH}/validated_documents"


def log_rejected_document(image_path, probability):
    file_exists = os.path.exists(REJECTED_LOG_PATH)
    with open(REJECTED_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_path", "manuscript_probability"])
        writer.writerow([image_path, probability])


def save_validated_document(image_path):
    os.makedirs(VALIDATED_IMAGES_DIR, exist_ok=True)
    img = cv2.imread(image_path)
    cv2.imwrite(os.path.join(VALIDATED_IMAGES_DIR, os.path.basename(image_path)), img)


def predict_manuscript(image_path, threshold=0.5):
    """Gate a user-submitted document: reject it here if it isn't manuscript-like
    before it reaches preprocessing, clustering or classification."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.uint8)[np.newaxis, ...] / 255.0

    probability = float(model.predict(x, verbose=0)[0][0])
    is_manuscript = probability >= threshold

    if not is_manuscript:
        log_rejected_document(image_path, probability)
    else:
        save_validated_document(image_path)

    return is_manuscript