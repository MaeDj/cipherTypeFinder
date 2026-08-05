import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

#Initial setup
DATASET_PATH = os.path.expanduser('~/Caramba/Dataset/corpus_cipherTypeFinder_Caramba')

#MODEL
#Training Config
BATCH_SIZE = 32
EPOCHS = 100
LATENT_DIM = 64   #Size of the feature vector
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = f"{DATASET_PATH}/preprocessing/processed/preprocessed_data.npy"
RESULTS_FOLDER = f"{DATASET_PATH}/clustering/hierarchical/clusters_{str(LATENT_DIM)}_silhouette_results"

#DEFINE DEEP AUTOENCODER
#This model learns to compress the image into a dense vector (Contrastive/Feature learning)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        #Encoder: Compresses 100x100 -> Latent Vector
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  #50x50
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), #25x25
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), #13x13
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 13 * 13, LATENT_DIM),
            nn.ReLU()
        )

        #Decoder: Reconstructs Latent Vector -> 100x100
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64 * 13 * 13),
            nn.ReLU(),
            nn.Unflatten(1, (64, 13, 13)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0), # 25x25 (approx)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 50x50
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 100x100
            nn.Sigmoid() #Output pixels 0-1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    #LOAD DATA
    print(f"Loading data from {DATA_PATH}...")
    try:
        data_dict = np.load(DATA_PATH, allow_pickle=True).item()
        images = data_dict['data']
        filenames = data_dict['filenames']
        print(f"Loaded {len(images)} images. Shape: {images.shape}")
    except FileNotFoundError:
        print("Error: preprocessed_data.npy not found. Run the preprocessing script first.")
        return

    filenames = np.array(filenames)

    #Normalize data for PyTorch (0-1 range) and add channel dimension
    #Images are (N, 100, 100) -> need (N, 1, 100, 100)
    X_tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0

    #Create DataLoader
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    #TRAIN MODEL
    print(f"\nTraining Autoencoder on {DEVICE}...")
    model = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            img = batch[0].to(DEVICE)

            #Forward
            reconstructed, _ = model(img)
            loss = criterion(reconstructed, img)

            #Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    #Plot training loss
    plt.figure()
    plt.plot(loss_history)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(RESULTS_FOLDER, "training_loss.png"))
    plt.close()

    #EXTRACT FEATURES
    print("\nExtracting latent features for clustering...")
    model.eval()
    features = []

    with torch.no_grad():
        #Process the full dataset in order (no shuffle) to match filenames
        full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        for batch in full_loader:
            img = batch[0].to(DEVICE)
            _, latent = model(img)
            features.append(latent.cpu().numpy())

    features = np.vstack(features)
    features = normalize(features, axis=1)
    print(f"Feature matrix shape: {features.shape}")

    #FIND OPTIMAL THRESHOLD
    print("\nSearching for optimal distance threshold...")
    #Test a range of thresholds based on common distances in latent space
    test_thresholds = np.linspace(0.5, 2, 16)
    best_score = -1
    best_threshold = float(test_thresholds[len(test_thresholds) // 2])

    scores = []
    for t in test_thresholds:
        cluster_test = AgglomerativeClustering(n_clusters=None, distance_threshold=t, linkage='ward')
        labels_test = cluster_test.fit_predict(features)

        n_found = len(set(labels_test))
        #Silhouette requires between 2 and N-1 clusters
        if 1 < n_found < len(features):
            score = silhouette_score(features, labels_test)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_threshold = t
            print(f"Threshold: {t:.1f} | Clusters: {n_found} | Silhouette: {score:.4f}")
        else:
            scores.append(0)

    if best_score == -1:
        print(f"Warning: no tested threshold produced a valid intermediate clustering (2..N-1 clusters); "
              f"falling back to the middle of the tested range: {best_threshold:.2f}")
    print(f"\nOptimization Complete. Best Threshold found: {best_threshold:.2f}")

    #Plot the Silhouette optimization curve
    plt.figure()
    plt.plot(test_thresholds, scores, marker='o')
    plt.title("Threshold Optimization (Silhouette Score)")
    plt.xlabel("Distance Threshold")
    plt.ylabel("Score (Higher is Better)")
    plt.savefig(os.path.join(RESULTS_FOLDER, "optimization_curve.png"))
    plt.close()

    #FINAL CLUSTERING
    final_clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=best_threshold,
        linkage='ward'
    )
    labels = final_clustering.fit_predict(features)
    n_clusters = len(set(labels))

    #FILTER & SAVE
    MIN_CLUSTER_SIZE = 5
    cluster_scores = {}
    valid_scores = []

    for label in range(n_clusters):
        indices = np.where(labels == label)[0]
        if len(indices) < MIN_CLUSTER_SIZE:
            cluster_scores[label] = 999.0
            continue

        cluster_feats = features[indices]
        centroid = np.mean(cluster_feats, axis=0)
        avg_dist = np.mean(np.linalg.norm(cluster_feats - centroid, axis=1))
        cluster_scores[label] = avg_dist
        valid_scores.append(avg_dist)

    #Variance thresholding
    var_threshold = np.percentile(valid_scores, 80) if valid_scores else 0

    #ORIGINAL CLASSIFICATION (unchanged logic, kept in its own folder for comparison)
    ORIGINAL_FOLDER = os.path.join(RESULTS_FOLDER, "Original")

    for label in range(n_clusters):
        indices = np.where(labels == label)[0]
        score = cluster_scores[label]
        size = len(indices)

        if size < MIN_CLUSTER_SIZE:
            path = os.path.join(ORIGINAL_FOLDER, "Rejected_TooSmall", f"Cluster_{label}")
        elif score > var_threshold:
            path = os.path.join(ORIGINAL_FOLDER, "Rejected_HighVar", f"Cluster_{label}")
        else:
            path = os.path.join(ORIGINAL_FOLDER, "Accepted", f"Cluster_{label}_Size{size}")

        os.makedirs(path, exist_ok=True)
        for idx in indices:
            cv2.imwrite(os.path.join(path, filenames[idx]), images[idx])

    #MERGED CLASSIFICATION: every character ends up under a single Accepted folder.
    #Too-small clusters keep their own subfolder there; High-Var clusters are
    #split character-by-character into whichever accepted cluster is closest.
    MERGED_FOLDER = os.path.join(RESULTS_FOLDER, "Merged")

    high_var_labels = []
    accepted_paths = {}
    accepted_centroids = {}

    for label in range(n_clusters):
        indices = np.where(labels == label)[0]
        score = cluster_scores[label]
        size = len(indices)

        if size >= MIN_CLUSTER_SIZE and score > var_threshold:
            high_var_labels.append(label)
            continue

        #Too-small or "clean" clusters are saved as their own accepted cluster
        path = os.path.join(MERGED_FOLDER, "Accepted", f"Cluster_{label}_Size{size}")
        os.makedirs(path, exist_ok=True)
        for idx in indices:
            cv2.imwrite(os.path.join(path, filenames[idx]), images[idx])

        if size >= MIN_CLUSTER_SIZE:
            accepted_paths[label] = path
            accepted_centroids[label] = np.mean(features[indices], axis=0)

    for label in high_var_labels:
        indices = np.where(labels == label)[0]
        for idx in indices:
            if accepted_centroids:
                best_label = min(
                    accepted_centroids,
                    key=lambda acc_label: np.linalg.norm(features[idx] - accepted_centroids[acc_label])
                )
                path = accepted_paths[best_label]
            else:
                #No accepted cluster exists to merge into: keep its own cluster folder
                #so the character still ends up under Accepted.
                path = os.path.join(MERGED_FOLDER, "Accepted", f"Cluster_{label}_Size{len(indices)}")
            os.makedirs(path, exist_ok=True)
            cv2.imwrite(os.path.join(path, filenames[idx]), images[idx])

    print(f"Done! Results saved in {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()