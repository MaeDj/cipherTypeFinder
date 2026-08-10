import cv2
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.utils.remote_listdir import listdir as remote_listdir
from model.utils.binarization_method import resolve_binarization_method
from model.utils.progress import progress
from model.utils.parallel import parallel_map_unordered
from model.utils.cleanup import cleanup_stage_output

#Initial setup
DATASET_PATH = os.path.expanduser('~/Caramba/Dataset/corpus_cipherTypeFinder_Caramba')

#Method forwarded as argv[1] when launched by main_preprocessing.py; falls back to an
#interactive prompt when run standalone.
bina_method = resolve_binarization_method(sys.argv[1] if len(sys.argv) > 1 else None)

input_folder = f"{DATASET_PATH}/preprocessing/cleaned/{bina_method}"
output_folder = f"{DATASET_PATH}/preprocessing/connectedComponent/{bina_method}/symbols"  #Cleaned symbols
boxes_folder = f"{DATASET_PATH}/preprocessing/connectedComponent/{bina_method}/bounding_boxes" #Bounding box visualization
os.makedirs(output_folder, exist_ok=True)
os.makedirs(boxes_folder, exist_ok=True)

#Modify parameter to fit the data
MIN_AREA = 75


#Was a hand-rolled pure-Python pixel-by-pixel flood fill (visited every pixel individually via a
#Python-level stack) - correct, but easily the single biggest contributor to this stage's runtime
#on full-size manuscript scans. cv2.connectedComponentsWithStats is the same union-find algorithm
#running in C, and returns identically-shaped output: stats rows are [x, y, w, h, area] with label 0
#reserved for background, matching what every call site here already expects.
def connected_components_with_stats(binary_image, connectivity=4):
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be either 4 or 8.")
    return cv2.connectedComponentsWithStats(binary_image, connectivity, cv2.CV_32S)

def _process_one(filename):
    """Extract character crops from a single cleaned line image, returning how many it produced.

    Runs in a worker process - relies only on module-level globals (input_folder/output_folder/
    boxes_folder/MIN_AREA) already set before the pool starts, which the default fork start
    method on Linux hands to every worker for free.

    Symbol filenames are `symbol_<doc_id>_<counter>` - doc_id already makes them unique across
    documents, so the counter only ever needs to be unique *within* one document's own characters
    (to preserve their reading-order sort in txtBuilder.py), not globally across the whole corpus.
    That means it can safely restart at 0 for every file instead of being one shared/incrementing
    global_counter threaded through the whole run - which is what let this loop become
    per-file, parallel work in the first place."""
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 0

    image_path = os.path.join(input_folder, filename)
    binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if binary_image is None:
        #A single unreadable/corrupt file must not abort the whole extraction pass.
        print(f"Warning: Could not read {image_path}, skipping", flush=True)
        return 0
    boxes_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    #Invert for connectedComponents (text=white, background=black)
    _, labels, stats, _ = connected_components_with_stats(255 - binary_image, connectivity=8)

    #Separate main symbols (area >= MIN_AREA) and diacritics (area < MIN_AREA)
    main_symbols = []
    diacritics = []
    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        if area >= MIN_AREA:
            main_symbols.append((i, x, y, w, h, area))
        else:
            diacritics.append((i, x, y, w, h, area))

    #Merge diacritics with their nearest main symbol (vertically aligned).
    #Each diacritic is assigned to only the closest aligned main symbol, not every aligned one.
    #Was an O(main_symbols x diacritics) Python double loop computing one center-distance at a
    #time; both are small lists of ints here, so numpy can build the whole distance matrix and
    #reduce it in a couple of vectorized calls instead.
    diacritic_best_match = {}
    if main_symbols and diacritics:
        symbol_centers_x = np.array([x + w // 2 for _, x, y, w, h, area in main_symbols])
        symbol_widths = np.array([w for _, x, y, w, h, area in main_symbols])
        diac_centers_x = np.array([dx + dw // 2 for _, dx, dy, dw, dh, d_area in diacritics])
        diac_widths = np.array([dw for _, dx, dy, dw, dh, d_area in diacritics])

        #distance[d, s] = |diac_center_x[d] - symbol_center_x[s]|; aligned[d, s] mirrors the
        #original per-pair "distance < max(w, dw) / 2" horizontal-alignment check.
        distance = np.abs(diac_centers_x[:, None] - symbol_centers_x[None, :])
        max_width = np.maximum(symbol_widths[None, :], diac_widths[:, None])
        aligned = distance < (max_width / 2)

        for diac_pos in range(len(diacritics)):
            aligned_symbol_positions = np.where(aligned[diac_pos])[0]
            if len(aligned_symbol_positions) == 0:
                continue
            #argmin returns the first minimum on ties, matching the original's strict "<" update
            #(later equal-distance candidates never overwrite an earlier one).
            best = aligned_symbol_positions[np.argmin(distance[diac_pos, aligned_symbol_positions])]
            diacritic_best_match[diac_pos] = (int(best), float(distance[diac_pos, best]))

    #Reverse-index once instead of re-scanning every match for every symbol (was another
    #O(symbols x diacritics) nested loop).
    diacritics_by_symbol = {}
    for diac_pos, (matched_symbol_pos, _) in diacritic_best_match.items():
        diacritics_by_symbol.setdefault(matched_symbol_pos, []).append(diacritics[diac_pos][0])

    merged_symbols = []
    for symbol_pos, symbol in enumerate(main_symbols):
        idx, x, y, w, h, area = symbol
        merged_indices = [idx] + diacritics_by_symbol.get(symbol_pos, [])

        #Merge bounding boxes
        x_min = min([x] + [stats[i][0] for i in merged_indices])
        y_min = min([y] + [stats[i][1] for i in merged_indices])
        x_max = max([x + w] + [stats[i][0] + stats[i][2] for i in merged_indices])
        y_max = max([y + h] + [stats[i][1] + stats[i][3] for i in merged_indices])
        new_w, new_h = x_max - x_min, y_max - y_min

        merged_symbols.append((merged_indices, x_min, y_min, new_w, new_h))

    #Save each merged symbol
    doc_id = os.path.splitext(filename)[0]
    local_counter = 0
    for indices, x, y, w, h in merged_symbols:
        #Create a mask combining all merged components
        merged_mask = np.zeros_like(binary_image, dtype=np.uint8)
        for idx in indices:
            merged_mask[labels == idx] = 255

        #Crop to bounding box and apply mask
        cropped_mask = merged_mask[y:y+h, x:x+w]
        cropped_original = binary_image[y:y+h, x:x+w]
        cleaned_symbol = cropped_original.copy()
        cleaned_symbol[cropped_mask == 0] = 255

        symbol_filename = os.path.join(output_folder, f"symbol_{doc_id}_{local_counter}.png")
        cv2.imwrite(symbol_filename, cleaned_symbol)
        local_counter += 1

        #raw bounding box (for visualization)
        cv2.rectangle(boxes_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #Save visualization with bounding boxes
    boxes_filename = os.path.join(boxes_folder, f"boxes_{filename}")
    cv2.imwrite(boxes_filename, boxes_image)

    return local_counter


if __name__ == "__main__":
    file_list = remote_listdir(input_folder)
    print(f"[connected_component:{bina_method}] {len(file_list)} documents to scan", flush=True)

    #Every document is scanned independently of every other one, so this fans out across a
    #process pool instead of one core handling the whole corpus alone (see model/utils/parallel.py).
    global_counter = 0
    for _filename, count in progress(parallel_map_unordered(_process_one, file_list),
                                      total=len(file_list), label=f"connected_component:{bina_method}"):
        global_counter += count

    #This is the character-crop count that drives the next stage's autoencoder batch count and,
    #more importantly, the O(n^2) silhouette_score search in hierarchical_silhouette_bucle.py -
    #worth watching for: tens of thousands is fine, well past ~50-80K gets slow/memory-heavy.
    print(f"[connected_component:{bina_method}] done - extracted {global_counter} character crops "
          f"from {len(file_list)} documents", flush=True)

    #Nothing downstream ever reads the cleaned line images again once characters have been
    #extracted from them - reclaim the space now instead of letting every stage's output pile
    #up on disk at once.
    cleanup_stage_output(input_folder, "cleaned line images")

    #The bounding-box visualizations are a debug/QA artifact only - no pipeline stage ever reads
    #them back, so unlike every other cleanup call here this one doesn't need to wait for a next
    #stage to "consume" them first.
    cleanup_stage_output(boxes_folder, "bounding-box visualizations")