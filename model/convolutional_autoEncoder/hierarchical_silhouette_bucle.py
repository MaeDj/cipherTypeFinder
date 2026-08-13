import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.utils.progress import stage, format_duration

#Initial setup
DATASET_PATH = os.path.expanduser('~/Caramba/Dataset/corpus_cipherTypeFinder_Caramba')

#MODEL
#Training Config
BATCH_SIZE = 32
EPOCHS = 100
LATENT_DIM = 64   #Size of the feature vector
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

#AgglomerativeClustering(ward).fit_predict and silhouette_score are both O(n^2) in the number of
#characters. Fine at a few thousand, but at the hundreds-of-thousands scale this pipeline now runs
#at, doing that twice per candidate threshold (16 candidates below) dwarfs every other stage
#combined. The search only has to pick *which* threshold looks best - a representative sample
#gives the same answer without the O(n^2) blowup. The final clustering after the search still
#runs on every character (that assignment is what actually gets saved to disk).
THRESHOLD_SEARCH_SAMPLE_SIZE = 8000

#The FINAL clustering below (the one whose labels actually get saved to disk) still has to run on
#every character, so sampling isn't an option there. Unconstrained ward linkage needs the full
#condensed pairwise-distance vector (n*(n-1)/2 float64s) in memory - at hundreds of thousands of
#characters that's hundreds of GB and reliably OOMs. Restricting merges to each point's k nearest
#neighbors (via sklearn's connectivity graph support) bounds that to O(n*k) instead, at the cost of
#only considering local merges - negligible for ward on a latent space this dense with THRESHOLD
#candidates already spanning 0.5-2.
FINAL_CLUSTERING_N_NEIGHBORS = 30

DATA_PATH = f"{DATASET_PATH}/preprocessing/processed/preprocessed_data.npy"
#Filenames now live next to the array as plain JSON rather than pickled together with
#it (see processing_for_clustering.py) -- keeps the array itself memory-mappable.
FILENAMES_PATH = f"{DATASET_PATH}/preprocessing/processed/preprocessed_filenames.json"
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


def _imwrite_checked(path, img):
    """cv2.imwrite fails silently (returns False, no exception) on disk-full/bad-path/unsupported
    format - which would otherwise let a full disk quietly produce a truncated cluster output while
    the pipeline still reports "Done!". This stage writes every character crop up to twice over
    (once under Original/, once under Merged/), so it's exactly the kind of place a quota limit
    gets hit; turn that into a loud, immediate failure instead of a silent one.

    cv2 is imported here rather than at module level so the rest of this module (model
    definition, GPU silhouette, training loop) stays importable - e.g. for benchmark_gpu.py -
    on any machine that has torch but not opencv."""
    import cv2
    if not cv2.imwrite(path, img):
        raise OSError(f"Failed to write image to {path} (disk full? invalid path?)")


def silhouette_score_gpu(features, labels, device=DEVICE):
    """Same definition as sklearn.metrics.silhouette_score (mean euclidean-distance silhouette,
    singleton clusters score 0), but vectorized on `device` instead of sklearn's single-threaded
    CPU implementation. The threshold search below calls this once per candidate every run, and
    the autoencoder already put a GPU in play for this pipeline - reusing it here (via a single
    (n, n) torch.cdist instead of sklearn's chunked CPU pairwise-distance loop) is effectively free
    at the sample sizes THRESHOLD_SEARCH_SAMPLE_SIZE keeps this to.
    """
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(labels, dtype=torch.int64, device=device)
    n = x.shape[0]
    idx = torch.arange(n, device=device)

    unique_labels = torch.unique(labels_t)
    own_cluster = torch.searchsorted(unique_labels, labels_t)  #labels_t re-indexed to 0..k-1

    dist = torch.cdist(x, x)  #(n, n) pairwise euclidean distances

    #One-hot cluster membership turns "sum of distances from each point to each cluster" into a
    #single (n, n) @ (n, k) matmul instead of a Python-level loop per cluster.
    one_hot = torch.zeros(n, unique_labels.numel(), device=device)
    one_hot[idx, own_cluster] = 1.0
    cluster_sizes = one_hot.sum(dim=0)  #(k,)
    dist_to_clusters = dist @ one_hot   #(n, k)

    own_size = cluster_sizes[own_cluster]
    #a(i): mean distance to own cluster, excluding the point itself
    a = dist_to_clusters[idx, own_cluster] / (own_size - 1).clamp(min=1)

    #b(i): mean distance to the nearest *other* cluster
    mean_dist_to_clusters = dist_to_clusters / cluster_sizes.clamp(min=1)
    mean_dist_to_clusters[idx, own_cluster] = float("inf")
    b = mean_dist_to_clusters.min(dim=1).values

    s = (b - a) / torch.maximum(a, b)
    s = torch.where(own_size <= 1, torch.zeros_like(s), s)  #sklearn convention: singleton -> 0
    return s.mean().item()


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    #LOAD DATA
    with stage("Loading preprocessed character data"):
        print(f"Loading data from {DATA_PATH}...", flush=True)
        try:
            images = np.load(DATA_PATH)
            with open(FILENAMES_PATH) as f:
                filenames = json.load(f)
            print(f"Loaded {len(images)} images. Shape: {images.shape}", flush=True)
        except FileNotFoundError:
            print("Error: preprocessed_data.npy or preprocessed_filenames.json not found. Run the preprocessing script first.", flush=True)
            return

        #The array and the filenames are now two separate files instead of one pickled unit
        #(see processing_for_clustering.py) - if they were ever produced by different runs, or one
        #got regenerated without the other, indices would silently line up with the wrong filename
        #instead of raising. Every crop written below is keyed by this pairing, so catch it here.
        if len(images) != len(filenames):
            print(f"Error: {len(images)} images but {len(filenames)} filenames - "
                  f"{DATA_PATH} and {FILENAMES_PATH} are out of sync. Re-run the preprocessing script.",
                  flush=True)
            return

    filenames = np.array(filenames)

    #Normalize data for PyTorch (0-1 range) and add channel dimension
    #Images are (N, 100, 100) -> need (N, 1, 100, 100)
    X_tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0

    #Create DataLoader
    #pin_memory + non_blocking .to() below let the host->device copy overlap with GPU compute
    #on the previous batch instead of stalling on every transfer; a no-op on CPU-only runs.
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=USE_CUDA)

    #TRAIN MODEL
    print(f"\nTraining Autoencoder on {DEVICE} - {EPOCHS} epochs, {len(dataloader)} batches/epoch...", flush=True)
    model = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    training_start = time.monotonic()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            img = batch[0].to(DEVICE, non_blocking=USE_CUDA)

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
        #Elapsed/ETA logged every epoch, not just every 10th - this loop is the first place
        #GPU-vs-CPU actually shows up as a live number instead of a guess.
        elapsed = time.monotonic() - training_start
        eta = elapsed / (epoch + 1) * (EPOCHS - epoch - 1)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}, "
                  f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)}", flush=True)
    print(f"Autoencoder training done in {format_duration(time.monotonic() - training_start)}", flush=True)

    #Plot training loss
    plt.figure()
    plt.plot(loss_history)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(RESULTS_FOLDER, "training_loss.png"))
    plt.close()

    #EXTRACT FEATURES
    extract_start = time.monotonic()
    print("\nExtracting latent features for clustering...", flush=True)
    model.eval()
    features = []

    with torch.no_grad():
        #Process the full dataset in order (no shuffle) to match filenames
        full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=USE_CUDA)
        for batch in full_loader:
            img = batch[0].to(DEVICE, non_blocking=USE_CUDA)
            _, latent = model(img)
            features.append(latent.cpu().numpy())

    features = np.vstack(features)
    features = normalize(features, axis=1)
    print(f"Feature matrix shape: {features.shape} - took {format_duration(time.monotonic() - extract_start)}",
          flush=True)

    #FIND OPTIMAL THRESHOLD
    #Search on a fixed-seed subsample (see THRESHOLD_SEARCH_SAMPLE_SIZE above) - both fit_predict
    #and the silhouette score below only need to rank thresholds against each other, not run on
    #every character. GPU silhouette per candidate keeps the O(n^2) part of even that sample on
    #the same device as the autoencoder instead of a single CPU core.
    test_thresholds = np.linspace(0.5, 2, 16)
    if len(features) > THRESHOLD_SEARCH_SAMPLE_SIZE:
        sample_idx = np.random.default_rng(42).choice(len(features), size=THRESHOLD_SEARCH_SAMPLE_SIZE, replace=False)
        search_features = features[sample_idx]
    else:
        search_features = features
    print(f"\nSearching for optimal distance threshold over {len(test_thresholds)} candidates "
          f"(sample of {len(search_features)} of {len(features)} characters, on {DEVICE})...", flush=True)
    best_score = -1
    best_threshold = float(test_thresholds[len(test_thresholds) // 2])

    threshold_search_start = time.monotonic()
    scores = []
    for i, t in enumerate(test_thresholds, 1):
        t_start = time.monotonic()
        cluster_test = AgglomerativeClustering(n_clusters=None, distance_threshold=t, linkage='ward')
        labels_test = cluster_test.fit_predict(search_features)

        n_found = len(set(labels_test))
        #Silhouette requires between 2 and N-1 clusters
        if 1 < n_found < len(search_features):
            score = silhouette_score_gpu(search_features, labels_test)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_threshold = t
            print(f"  [{i}/{len(test_thresholds)}] Threshold: {t:.1f} | Clusters: {n_found} | "
                  f"Silhouette: {score:.4f} | took {format_duration(time.monotonic() - t_start)}", flush=True)
        else:
            scores.append(0)
            print(f"  [{i}/{len(test_thresholds)}] Threshold: {t:.1f} | Clusters: {n_found} "
                  f"(skipped, outside [2, N-1]) | took {format_duration(time.monotonic() - t_start)}", flush=True)
    print(f"Threshold search done in {format_duration(time.monotonic() - threshold_search_start)}", flush=True)

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
    #Unlike the threshold search above, this has to run on every character - so it can't fall back
    #to sampling. Plain ward linkage needs the full (n, n) pairwise-distance matrix, which is the
    #700GB+ allocation that OOMs at this dataset's scale (see FINAL_CLUSTERING_N_NEIGHBORS above).
    #A k-nearest-neighbor connectivity graph keeps merges local and memory at O(n*k).
    final_start = time.monotonic()
    n_neighbors = min(FINAL_CLUSTERING_N_NEIGHBORS, len(features) - 1)
    print(f"\nBuilding {n_neighbors}-nearest-neighbor connectivity graph for the full-dataset "
          f"clustering ({len(features)} characters)...", flush=True)
    connectivity = kneighbors_graph(features, n_neighbors=n_neighbors, include_self=False)

    final_clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=best_threshold,
        linkage='ward',
        connectivity=connectivity
    )
    labels = final_clustering.fit_predict(features)
    n_clusters = len(set(labels))
    print(f"Final clustering: {n_clusters} clusters - took {format_duration(time.monotonic() - final_start)}",
          flush=True)

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
    original_save_start = time.monotonic()
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
            _imwrite_checked(os.path.join(path, filenames[idx]), images[idx])
    print(f"Original classification saved ({len(features)} crops) - "
          f"took {format_duration(time.monotonic() - original_save_start)}", flush=True)

    #MERGED CLASSIFICATION: every character ends up under a single Accepted folder.
    #Too-small clusters keep their own subfolder there; High-Var clusters are
    #split character-by-character into whichever accepted cluster is closest.
    merged_save_start = time.monotonic()
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
            _imwrite_checked(os.path.join(path, filenames[idx]), images[idx])

        if size >= MIN_CLUSTER_SIZE:
            accepted_paths[label] = path
            accepted_centroids[label] = np.mean(features[indices], axis=0)

    #Nearest-accepted-centroid assignment for every high-variance character. Was one
    #np.linalg.norm() python-level call per (character, accepted cluster) pair - for H
    #high-variance characters and K accepted clusters that's H*K individual numpy calls, each
    #paying Python/numpy dispatch overhead for a handful of FLOPs. Computed here instead as one
    #(H, K) squared-distance matrix via the ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b expansion, so the
    #dominant a.b term is a single BLAS matrix multiply - avoids materializing an (H, K, D) tensor
    #while still giving the same nearest-centroid result (squared distance preserves the argmin).
    if accepted_centroids:
        acc_labels = list(accepted_centroids.keys())  #dict insertion order, matches min()'s tie-breaking below
        centroid_matrix = np.stack([accepted_centroids[acc_label] for acc_label in acc_labels])

    for label in high_var_labels:
        indices = np.where(labels == label)[0]
        if accepted_centroids:
            char_feats = features[indices]
            sq_dist = (
                np.sum(char_feats ** 2, axis=1)[:, None]
                + np.sum(centroid_matrix ** 2, axis=1)[None, :]
                - 2 * char_feats @ centroid_matrix.T
            )
            best_positions = np.argmin(sq_dist, axis=1)  #first minimum on ties, matching the original min()
            for idx, best_pos in zip(indices, best_positions):
                path = accepted_paths[acc_labels[best_pos]]
                os.makedirs(path, exist_ok=True)
                _imwrite_checked(os.path.join(path, filenames[idx]), images[idx])
        else:
            #No accepted cluster exists to merge into: keep its own cluster folder
            #so the character still ends up under Accepted.
            path = os.path.join(MERGED_FOLDER, "Accepted", f"Cluster_{label}_Size{len(indices)}")
            os.makedirs(path, exist_ok=True)
            for idx in indices:
                _imwrite_checked(os.path.join(path, filenames[idx]), images[idx])
    print(f"Merged classification saved - took {format_duration(time.monotonic() - merged_save_start)}", flush=True)

    print(f"Done! Results saved in {RESULTS_FOLDER}", flush=True)


if __name__ == "__main__":
    main()