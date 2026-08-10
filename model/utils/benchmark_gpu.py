###Claude suggested helper###
###
#Standalone timing tool for the two costliest parts of stage 2 (convolutional_autoEncoder/
#hierarchical_silhouette_bucle.py): autoencoder training/feature-extraction (GPU) and
#AgglomerativeClustering(ward) (CPU, O(n^2), no GPU implementation in this dependency stack).
#Instead of guessing from FLOP counts, this measures real throughput on the actual machine and
#extrapolates:
#  - autoencoder: times real batches on DEVICE, scales linearly to EPOCHS * n_images
#  - clustering: fit_predict is O(n^2), so timing it at a few small n and fitting a quadratic
#    curve gives an extrapolated estimate at the real n without waiting hours for it to finish
#
#Usage:
#  python -m model.utils.benchmark_gpu
#  python -m model.utils.benchmark_gpu --n-images 433365 --train-batches 200
#  python -m model.utils.benchmark_gpu --clustering-probe-sizes 2000 4000 8000 16000
###
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.convolutional_autoEncoder.hierarchical_silhouette_bucle import (
    BATCH_SIZE, DEVICE, EPOCHS, DATA_PATH, ConvAutoencoder, silhouette_score_gpu,
)
from model.utils.progress import format_duration


def _infer_n_images(default):
    """Use the real preprocessed dataset's size if it's present on this machine, otherwise fall
    back to the caller-supplied estimate - keeps the tool useful before stage 1 has ever run."""
    try:
        return int(np.load(DATA_PATH, mmap_mode='r').shape[0])
    except (FileNotFoundError, OSError):
        return default


def benchmark_autoencoder(n_images, train_batches, warmup_batches=20):
    """Times real forward+backward+optimizer-step batches on DEVICE with the exact model/batch
    size/loss the pipeline trains with, then scales that measured rate to the full run
    (EPOCHS * n_images for training, one more pass over n_images for feature extraction)."""
    model = ConvAutoencoder().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #Synthetic data: this architecture's per-batch cost doesn't depend on pixel content, only on
    #shape/dtype/device, so random noise measures the same throughput as the real corpus would.
    batch = torch.rand(BATCH_SIZE, 1, 100, 100, device=DEVICE)

    def _train_step():
        reconstructed, _ = model(batch)
        loss = criterion(reconstructed, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(warmup_batches):
        _train_step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.monotonic()
    for _ in range(train_batches):
        loss = _train_step()
        loss.item()  #mirrors the pipeline's per-batch host sync (see hierarchical_silhouette_bucle.py)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    train_elapsed = time.monotonic() - start
    train_batches_per_sec = train_batches / train_elapsed

    #Feature extraction: forward-only, no grad - timed separately since it's meaningfully cheaper.
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_batches):
            model(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.monotonic()
        for _ in range(train_batches):
            model(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    extract_elapsed = time.monotonic() - start
    extract_batches_per_sec = train_batches / extract_elapsed

    batches_per_epoch = -(-n_images // BATCH_SIZE)  #ceil
    training_time = (batches_per_epoch * EPOCHS) / train_batches_per_sec
    extraction_time = batches_per_epoch / extract_batches_per_sec

    print(f"Device: {DEVICE}")
    print(f"Measured training rate:   {train_batches_per_sec:6.1f} batches/s "
          f"({train_batches_per_sec * BATCH_SIZE:8.1f} images/s) over {train_batches} timed batches")
    print(f"Measured extraction rate: {extract_batches_per_sec:6.1f} batches/s "
          f"({extract_batches_per_sec * BATCH_SIZE:8.1f} images/s) over {train_batches} timed batches")
    print(f"Extrapolated to n_images={n_images}, {EPOCHS} epochs, batch={BATCH_SIZE}:")
    print(f"  Training:           {format_duration(training_time)}")
    print(f"  Feature extraction: {format_duration(extraction_time)}")
    print(f"  Stage 2a total:     {format_duration(training_time + extraction_time)}")
    return training_time + extraction_time


def benchmark_clustering(n_images, probe_sizes, latent_dim=64):
    """Ward-linkage AgglomerativeClustering has no GPU implementation in this project's
    dependency stack (sklearn only) and is O(n^2) - too slow to just run once at the real n to
    see how long it takes. Instead, time it at a handful of small n and fit time = a*n^2 + b to
    extrapolate to the real n_images without waiting for it to finish there."""
    rng = np.random.default_rng(0)
    probe_sizes = sorted(p for p in probe_sizes if p < n_images) or [min(probe_sizes)]

    print(f"\nClustering probe (ward linkage, {latent_dim}-dim features, random data - "
          f"AgglomerativeClustering's cost depends on n and dimensionality, not on the data "
          f"itself):")
    measured = []
    for n in probe_sizes:
        x = rng.normal(size=(n, latent_dim)).astype(np.float32)
        start = time.monotonic()
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0, linkage='ward').fit_predict(x)
        elapsed = time.monotonic() - start
        measured.append((n, elapsed))
        print(f"  n={n:7d}  took {format_duration(elapsed)}  ({elapsed:.3f}s)")

        sil_start = time.monotonic()
        silhouette_score_gpu(x, labels)
        sil_elapsed = time.monotonic() - sil_start
        print(f"    silhouette_score_gpu on same n: {format_duration(sil_elapsed)} ({sil_elapsed:.3f}s) on {DEVICE}")

    #Least-squares fit of elapsed = a*n^2 + b*n (ward's dominant cost is quadratic; the linear
    #term absorbs fixed overhead) using the probe points actually measured above.
    ns = np.array([n for n, _ in measured], dtype=np.float64)
    ts = np.array([t for _, t in measured], dtype=np.float64)
    coeffs = np.polyfit(ns, ts, deg=2) if len(measured) >= 3 else np.polyfit(ns, ts, deg=1, full=False).tolist() + [0.0]
    predicted = np.polyval(coeffs, n_images)

    print(f"\nQuadratic fit from {len(measured)} probe points extrapolated to n_images={n_images}:")
    print(f"  Estimated single AgglomerativeClustering(ward).fit_predict: {format_duration(predicted)}")
    print(f"  (This runs once per search-loop candidate on the {min(n_images, 8000)}-point sample "
          f"during the threshold search, and once more on the full {n_images} for the final "
          f"clustering - see THRESHOLD_SEARCH_SAMPLE_SIZE in hierarchical_silhouette_bucle.py.)")
    return predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-images", type=int, default=None,
                         help="Character count to extrapolate to. Defaults to the real "
                              "preprocessed_data.npy size if found, else 433365.")
    parser.add_argument("--train-batches", type=int, default=100,
                         help="Number of real batches to time for the autoencoder rate (default 100).")
    parser.add_argument("--clustering-probe-sizes", type=int, nargs="+", default=[2000, 4000, 8000],
                         help="Sample sizes to actually run AgglomerativeClustering at, used to "
                              "fit the O(n^2) extrapolation (default 2000 4000 8000).")
    parser.add_argument("--skip-clustering", action="store_true",
                         help="Skip the clustering probe (autoencoder timing only).")
    args = parser.parse_args()

    n_images = args.n_images if args.n_images is not None else _infer_n_images(433365)

    stage2a_time = benchmark_autoencoder(n_images, args.train_batches)

    if not args.skip_clustering:
        clustering_time = benchmark_clustering(n_images, args.clustering_probe_sizes)
        print(f"\nStage 2 (autoencoder + one full clustering pass) rough total: "
              f"{format_duration(stage2a_time + clustering_time)}")
        print("Note: the threshold search still runs up to 16 clustering + silhouette calls on "
              "the (small, fast) sample on top of that - see stage output for their live timing.")


if __name__ == "__main__":
    main()