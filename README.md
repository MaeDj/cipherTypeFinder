# CipherTypeFinder
June 2026

Program created for Caramba Team, Loria.

The aim of this program is to get from a raw photo of a (supposedly) ciphered manuscript, its cipher type included into 7 categories.

## Pipeline Overview

| # | Step | Status |
|---|------|--------|
| 0 | Manuscript detection (CNN) | 🚧 Architecture defined (`model/manuscript_detection/CNN.py`), corpus not yet built |
| 1 | Image & data preprocessing | ✅ Implemented (`model/preprocessing`) |
| 2 | Character clustering (convolutional autoencoder) | ✅ Implemented (`model/convolutional_autoEncoder`) |
| 3 | Computable text reconstruction | ✅ Implemented (`model/txt_equivalent_builder`) |
| 4 | Manuscript statistics (alphabet size, index of coincidence, symbol frequencies) | ✅ Implemented (`model/statistics`) |
| 5 | Cipher type classification (feature vectors + neural network) | 🚧 Tabular-only baseline implemented (`model/text_clusterization/MLP_cipher_classifier.py`); full image+tabular fusion architecture still just defined |
| — | User-provided document info (origin, date, plaintext language, character type) | ✅ Implemented (`model/user_interface/user_caller.py`) |

---

## Running the Pipeline & Server Notes

`model/main.py` is the single entry point (`cd model && python3 main.py`) and runs every stage below in order, in-process. Since it's routinely run on a different machine than it's developed on (typically `biscotte`, the Caramba lab server holding the corpus), a few things are handled specifically so paths and subprocess calls stay correct regardless of which host or working directory it's launched from:

- **Dataset location is a single absolute path, not a repo-relative one.** Every stage resolves its data through the same constant, expanded once per script: `DATASET_PATH = os.path.expanduser('~/Caramba/Dataset/corpus_cipherTypeFinder_Caramba')`.  — `main.py` imports every stage module up front, so a cwd-relative path would resolve differently (or to nothing) depending on where `python3 main.py` was launched from. Every generated artifact (binarized images, line/character crops, clustering results, computable `.txt`/`.csv`, statistics) is written under this same tree — `<DATASET_PATH>/preprocessing/...`, `<DATASET_PATH>/clustering/...`, `<DATASET_PATH>/computable/...` — alongside the raw `original_corpus/{img,metadata}` it reads from. It's kept outside the git repo (multi-GB of images/generated data); whichever host runs the pipeline needs this directory populated at that exact path, or the constant updated to match.

- **Local-first, SFTP-fallback directory listing** (`model/utils/remote_listdir.py`). `remote_listdir.listdir()` stands in for `os.listdir()` everywhere in the pipeline. It checks the local filesystem first and only falls back to SFTP if the path genuinely isn't reachable locally — so running directly on the machine that holds `DATASET_PATH` (e.g. on `biscotte` itself) needs no SFTP configuration at all. Running from elsewhere (e.g. a laptop without the corpus) falls back to an SFTP connection, configured through a `.env` file at the **project root** (`CipherTypeFinder/.env`, one level above `model/`) holding `SFTP_HOST`, `SFTP_PORT`, `SFTP_USERNAME`, `SFTP_PASSWORD`. `.env` is git-ignored, so it has to be created by hand on every machine that needs the SFTP path.

- **One binarization-method prompt for the whole run**, not one per stage. `main.py` asks once at the very start (`resolve_binarization_method()`, `model/utils/binarization_method.py`) and forwards the chosen method to every stage that needs it, as a plain CLI argument to the `preprocessing/` scripts (each launched as its own subprocess by `main_preprocessing.py`). Each of those scripts still falls back to its own interactive prompt if run standalone (e.g. `python3 ./binarize.py` with no argument) — the prompt/default logic itself lives once in `binarization_method.py` rather than being duplicated per file.

- **Portable subprocess launching** (`main_preprocessing.py`). Preprocessing stages 1–5 each run as a subprocess (`subprocess.run([sys.executable, ...], check=True)`). Two details make this work regardless of host/cwd: scripts are launched by their absolute path (`Path(__file__).resolve().parent / "<script>.py"`, anchored to the launcher's own location) rather than `./<script>.py`; and `sys.executable` is used instead of a hardcoded `python3`, so each stage runs under the same interpreter/virtualenv as `main.py` itself (relevant on servers where a bare `python3` on `PATH` may not be the intended environment, e.g. `envCipherTypeFinder`). `check=True` also means a failing stage now raises instead of being silently swallowed.

- **Import-safe stage modules.** `txtBuilder.py`, `equivalent_txt_manuscripts_stats.py`, `MLP_cipher_classifier.py` and `hierarchical_silhouette_bucle.py` each guard their execution with `if __name__ == "__main__": main()`. `main.py` imports all of them up front; without the guard, each one's full pipeline (and any early `exit()`/`return` on missing input) would fire during import — before earlier stages had produced the output they depend on — rather than when `main.py` actually calls them in sequence.

---

## User-Provided Document Info (`model/user_interface`)

Before (or alongside) the automated pipeline, `user_caller.py` lets a user attach expert knowledge about their own document that cannot be inferred from the image alone:

1. **`prompt_folder_path`** — asks for the folder containing the user's document, re-prompting until it points to an existing directory.
2. **`read_existing_info`** — loads a previously saved `document_info.json` from that folder, if any, so the user can revisit and edit prior answers instead of starting blank.
3. **`collect_document_info`** — interactively gathers four optional fields, each of which can be left/set to `None`:
   - **Origin**
   - **Date**
   - **Plain text language** (useful for interpreting the index of coincidence)
   - **Character type** 
4. **`write_info`** — persists the collected answers back into `document_info.json` inside the document's folder.

Origin, date and plaintext language are no longer hardcoded gaps, they are now user-suppliable inputs that the classifier below knows how to consume.

---

## 0. Manuscript Detection (`model/manuscript_detection`)

This is the **gatekeeper of the whole pipeline**: before any preprocessing, clustering, or cipher-type analysis happens, a binary CNN (`manuscript_detection_CNN.py`) decides whether a raw input image actually looks like a manuscript at all. Any image a user submits that is **not** manuscript-like is rejected here and never reaches the rest of the pipeline — it does not get routed into the "Not encrypted / not a manuscript" cipher-type class from step 5, which only applies to manuscripts that passed this gate but turned out to be unciphered/unreadable.

1. **Dataset loading** (`load_custom_dataset`) — reads a corpus of manuscript vs. random images (`<DATASET_PATH>/corpus_manuscript_random/{images,labels}`, not yet built — see [Running the Pipeline & Server Notes](#running-the-pipeline--server-notes) for what `DATASET_PATH` resolves to), resizing every image to the same fixed 100×100 canvas used by the rest of the project (see `model/preprocessing/processing_for_clustering.py`), and split 80/20 into train/test with `train_test_split`.
2. **Architecture** — a small stack of `Conv2D` + `MaxPooling2D` + `Dropout` blocks (32 → 64 → 64 → 128 filters) followed by a `Flatten` → `Dense(64, relu)` → `Dropout` → `Dense(1, sigmoid)` output, trained with binary cross-entropy / Adam.
3. **Output** — a single probability that the input image is a manuscript; images falling below the decision threshold are rejected before entering the preprocessing stage below. Rejected images are logged to `rejected_documents.csv` (`log_rejected_document`) with their probability for later review, while validated images are copied into the `validated_documents/` folder (`save_validated_document`) via `cv2.imwrite`.

---

## 1. Preprocessing (`model/preprocessing`)

Once an image is confirmed to be a manuscript, it goes through the preprocessing pipeline, orchestrated end-to-end by `main_preprocessing.py`. The overall approach is adapted from https://dspace.ut.ee/server/api/core/bitstreams/e8c1cae6-cc96-43e8-993b-0f4f29f5312d/content.

1. **Binarization** (`binarize.py`) — converts the grayscale scan to black & white text. Six interchangeable methods are available (Otsu, Gaussian-Otsu, Adaptive, Niblack, Sauvola, Local min/max), selectable at runtime, with Sauvola as the default.
2. **Line segmentation** (`line_segmentation.py`) — computes a horizontal projection profile of the binarized page to detect text-line peaks, assigns each connected component to its nearest line, and reconstructs/saves each text line as its own image.
3. **Cleaning** (`cleaning.py`) — fits a linear regression through the foreground pixels of each line to estimate its baseline/middle zone, then removes connected components that touch the top/bottom border without reaching that middle zone (page artifacts, bleed-through, noise) as well as components below a minimum area.
4. **Connected component analysis** (`connected_component.py`) — extracts individual characters from each cleaned line via 8-connectivity connected components. Small components (diacritics, dots, accents) are merged into the nearest horizontally-aligned main glyph so composite characters stay intact. Each resulting character is saved as its own cropped image, with bounding-box visualizations kept for inspection.
5. **Processing for clustering** (`processing_for_clustering.py`) — resizes every extracted character to a fixed 100×100 canvas (aspect-ratio preserved, white-padded) and stacks the whole set into a single `.npy` array, ready to feed the autoencoder.

This produces:
- **Document characters**: individual character images
---

## 2. Character Clustering (`model/convolutional_autoEncoder`)

Implements the character-clustering approach described in *"Unsupervised Feature Learning via Convolutional Autoencoders for Cross-Manuscript Comparison in Historical Cryptanalysis"* (Alejandra Reinares, Giuseppe De Gregorio, Alicia Fornés). A cluster groups together all the characters across the corpus that look alike (i.e. are likely the same underlying glyph/symbol).

`hierarchical_silhouette_bucle.py`:

1. Loads the 100×100 preprocessed character images produced by the preprocessing step.
2. Trains a convolutional autoencoder (3-layer convolutional encoder down to a 64-dimensional latent vector, mirrored transposed-convolutional decoder) to reconstruct each character image (MSE loss, Adam optimizer).
3. Extracts and L2-normalizes the latent vector of every character from the trained encoder — this is the character's feature representation.
4. Searches for the optimal distance threshold for Agglomerative (ward-linkage) clustering by scanning a range of thresholds and keeping the one maximizing the silhouette score.
5. Runs the final hierarchical clustering with that threshold, then classifies each resulting cluster by size (<5 members = too small) and by average intra-cluster distance (top 20% = too high-variance), and saves the outcome under two parallel folders for comparison:
   - **`Original/`** — the straightforward classification: too-small and too-high-variance clusters are set aside as "Rejected" (`Rejected_TooSmall`, `Rejected_HighVar`), while the rest are kept as "Accepted" clusters, each ideally corresponding to a single distinct character/glyph.
   - **`Merged/`** — a lossless variant where no character is discarded: too-small clusters are kept as their own accepted cluster, and every character from a too-high-variance cluster is individually reassigned to whichever accepted cluster's centroid is closest, so all characters end up under a single `Accepted` folder. (Necessary to build equivalent alphabet)

---

## 3. Computable Text Reconstruction (`model/txt_equivalent_builder`)

Once each character has been assigned to a cluster, each manuscript document is reconstructed as a computable `.txt` equivalent: every character occurrence is replaced, in its original reading order, by a simple, easily-computable symbol representing its cluster (rather than the raw glyph image). This turns each manuscript into plain text that downstream numerical/statistical analysis can work with directly.

`txtBuilder.py`:

1. Lists every document ID present in the original corpus image folder (`<DATASET_PATH>/original_corpus/img`).
2. For each document, walks the "Accepted" clusters produced by the character-clustering step and, by matching each character image's filename (`symbol_<doc>_<global_counter>`), links every character back to its document and cluster label, keeping the character's global position counter.
3. Sorts each document's characters by that position counter to restore the original reading order.
4. Writes, per document, a plain `.txt` file containing the space-separated sequence of cluster labels (`<DATASET_PATH>/computable/text/<doc>.txt`), plus a companion `.csv` recording each character's cluster label alongside its source image filename (`<DATASET_PATH>/computable/csv/<doc>.csv`) for traceability back to the glyph images.

---

## 4. Manuscript Statistics (`model/statistics`)

Once the computable `.txt` equivalents exist, each document is run through a statistical analysis that turns its symbol sequence into the numerical features consumed by the classification model in step 5.

`equivalent_txt_manuscripts_stats.py`:

1. Lists every document ID present in `<DATASET_PATH>/computable/text` and splits each document's `.txt` equivalent back into its ordered list of symbols.
2. For each document, computes:
   - **Alphabet size** — number of distinct symbols used in the document.
   - **Index of coincidence difference** (`coincidence_index`) — the document's actual index of coincidence (probability that two symbols drawn at random from the document are identical) minus the expected index of coincidence for its stated plaintext language(s) (`plaintext_lang`, tokenized to tolerate free text like `"French? Portuguese?"` and averaged across every recognized language listed). The reference values used for that expectation come from the `COINCIDENCE_INDEX_CORRESPONDENCE` constant, a `{language: expected_IC}` lookup table hardcoded with the published index of coincidence for each of the 7 plaintext languages supported by the corpus (French 0.0778, English 0.0667, Spanish 0.0770, Portuguese 0.0745, German 0.0762, Latin 0.0765, Italian 0.0738) — the same language set enforced upstream by `is_allowed_plaintext_lang` in `fetch_data.py`. `None`/empty when `plaintext_lang` names no recognized language (i.e. no key of `COINCIDENCE_INDEX_CORRESPONDENCE` matches), so the raw index alone can't be compared meaningfully.
   - **Symbol frequencies** — occurrence count and relative frequency of every symbol.
3. Registers all documents into a single `.csv` (`<DATASET_PATH>/computable/statistics/manuscripts_stats.csv`): one row per document (`doc_id`, `total_symbols`, `alphabet_size`, `coincidence_index`), the DECODE metadata carried over from `fetch_data.py` (`origin`, `start_year`, `cipher_types`, `symbol_sets`), followed by one `freq_<symbol>` column per symbol found anywhere in the corpus (0 where a document doesn't use that symbol) — so every document row shares the same columns while metadata and index of coincidence are each stored only once per document. `plaintext_lang` itself is deliberately excluded from the `.csv`: it's only used internally to compute `coincidence_index`.

---

## 5. Cipher Type Classification

### 5.1 Implemented Baseline: Tabular MLP (`model/text_clusterization/MLP_cipher_classifier.py`)

A first baseline is implemented and runnable end-to-end: a multi-label `MLPClassifier` (scikit-learn, see [neural_networks_supervised](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)) trained purely on the tabular statistics from step 4 — it does not yet use the manuscript image data described in the target architecture below.

- **Input**: `manuscripts_stats.csv` (step 4's output). Features used:
  - **Numeric**: `total_symbols`, `alphabet_size`, `coincidence_index`, `start_year` and every `freq_<symbol>` column, with missing values imputed to the column median (`numeric_medians`). `coincidence_index` already folds in the document's `plaintext_lang` at step 4 (it's the difference against that language's expected index of coincidence), which is why `plaintext_lang` itself isn't re-encoded as a separate feature here.
  - **Origin**: one-hot encoded (`OneHotEncoder(handle_unknown='ignore')`), so an unseen/missing origin at prediction time is safely encoded as all-zero rather than erroring.
  - **Symbol sets** (`symbol_sets`): comma-separated codes parsed with `split_codes` and multi-label binarized (`symbol_set_binarizer`).
  
  All three groups are concatenated into a single feature matrix (`load_dataset`), and every fitted encoder is returned alongside it so a brand-new document — described only by the user through `user_caller.py` — can be encoded the same way via `build_feature_row` before being passed to `predict_cipher_types`. Every field besides the base tabular statistics is optional: a missing origin or symbol set degrades gracefully instead of blocking prediction.
- **Labels**: `cipher_types`, as recorded by the DECODE Records Database (see Data Acquisition below), is a comma-separated field — a manuscript can combine several cipher types at once — making this a genuine multi-label problem rather than multi-class. `MultiLabelBinarizer` turns it into one binary column per class:

  | Code | Class |
  |------|-------|
  | 1 | Monoalphabetic |
  | 2 | Homophonic |
  | 3 | Machine |
  | 4 | Polyalphabetic |
  | 5 | Nomenclature |
  | 7 | Polyphonic |

- **Model**: `StandardScaler` → `MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', early_stopping=True)`. Since the target is a 2D binary matrix rather than a single label column, scikit-learn puts one sigmoid unit per class on the output layer instead of a single softmax, natively supporting multi-label prediction and per-class probabilities (`predict_proba`).
- **Evaluation**: per-class `classification_report`, exact-match accuracy, Hamming loss, and samples-averaged Jaccard score — plain accuracy is misleading for multi-label problems, so these are reported instead.

### 5.2 Target Architecture: Image + Tabular Fusion (planned, not yet implemented)

The intended final model goes further than the tabular baseline above by also feeding in the manuscript's image patches, fusing both vector types before classification.
This upgrade is not already available and ask for an image preprocessing + image patching + image encoder to get image vectorial representation.

#### 5.2.1 Feature Vectors

Outputs of preprocessing/clustering feed into two vector types:

- **Vector Numerical Data**, made from:
  - The tabular data made in 5.1 with information that were deduced from a statistic analysis of computable text equivalent of the manuscript (see step 4, `model/statistics`). This is the vector currently consumed alone by the 5.1 baseline.
  - UPGRADE NOT ALREADY AVAILABLE: cut off step 5.1 to implement instead a tabular data encoder and merge image and tabular data vector.
  - 
  These features were defined from expert knowledge represented in this ontology-like representation:

  <img src="./schema_ontologie_systeme_chiffremen.png" alt="shema_ontology_cipher_type"/>

- **Vector Image**, 
UPGRADE NOT ALREADY AVAILABLE: made from the manuscript's image patches.

#### 5.2.2 Neural Network Architecture


UPGRADE NOT ALREADY AVAILABLE:

**Fusion**
=> Flatten Layer
- **TIP or MMCL — MERGE** (combine image vector and tabular vector) (arXiv:2407.07582v1 [cs.CV] 10 Jul 2024)

**Processing**
**Recurrent Neural Network**
=> Hidden Layer
- Activation function: **ReLU**

**Finalization**
=> Output layer
- Activation function: **SoftMax**
- Loss Function: **Triplet Loss**
- Backpropagation function / Optimization: **Adam**
- Evaluation metric: **Precision**

#### 5.2.3 Output: Cipher Type (Classification)

- Not encrypted / not a manuscript
- Machine
- Homophonic
- Nomenclature
- Polyalphabetic
- Polyphonic
- Monoalphabetic

Each class outputs a **Probability P**. Multiple-class predictions can also be considered; experts will define a threshold to decide which predictions are acceptable.



---

<img src="./shema_architecture_cipherTypeFinder.png" alt="shema_architecture_cipherTypeFinder" style="zoom:200%;" />

---

## Data Acquisition

Input ciphered documents (`.jpg`) come from the DECODE Records Database of the DE-CRYPT Project. `model/corpus_builder/fetch_data.py` logs into the DE-CRYPT API, lists eligible records — excluding cipher type `6` (undetermined) and any record whose `plaintext_lang` isn't confidently one of the currently supported languages (French, Portuguese, German, Spanish, Latin, Italian, English; see `is_allowed_plaintext_lang`) — and downloads their images and metadata (origin region/city, start year, cipher types, plaintext language, symbol sets) into a local corpus folder.

## AI Usage Declaration

This project was partially built using Claude Code (https://claude.ai).

Please find below the list of tasks performed using AI:

- Code verification (path and call unicity / compilation checks)
- Server-specific issues (see Running the Pipeline & Server Notes)
- Automation of repetitive implementation fixes (replacing incorrect structures / incorrect path calls)
- Organization and standardization of the documentation (based on human instructions, structure, and reasoning)