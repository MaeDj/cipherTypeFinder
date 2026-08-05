import cv2
import numpy as np
import os
import pdb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.utils.remote_listdir import listdir as remote_listdir
from model.utils.binarization_method import resolve_binarization_method
from model.utils.progress import progress

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


def connected_components_with_stats(binary_image, connectivity=4):
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be either 4 or 8.")
    
    rows, cols = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)
    labels = np.zeros_like(binary_image, dtype=np.int32)

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

    current_label = 1  #Label 0 will be background
    stats = []
    centroids = []

    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 255 and not visited[i, j]:
                #Begin new component
                coords = []
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if not (0 <= x < rows and 0 <= y < cols):
                        continue
                    if visited[x, y] or binary_image[x, y] != 255:
                        continue
                    visited[x, y] = True
                    labels[x, y] = current_label
                    coords.append((x, y))
                    for dx, dy in neighbors:
                        stack.append((x + dx, y + dy))

                coords_np = np.array(coords)
                ys, xs = coords_np[:, 0], coords_np[:, 1]
                x_min, y_min = xs.min(), ys.min()
                x_max, y_max = xs.max(), ys.max()
                width = x_max - x_min + 1
                height = y_max - y_min + 1
                area = len(coords)
                cx = xs.mean()
                cy = ys.mean()

                stats.append([x_min, y_min, width, height, area])
                centroids.append([cx, cy])
                current_label += 1

    #Add background as label 0
    stats = [[0, 0, 0, 0, 0]] + stats
    centroids = [[0.0, 0.0]] + centroids

    return current_label, labels, np.array(stats, dtype=np.int32), np.array(centroids, dtype=np.float32)

#Tracking total num of characters across all files
global_counter = 0

file_list = remote_listdir(input_folder)
print(f"[connected_component:{bina_method}] {len(file_list)} documents to scan", flush=True)

for filename in progress(file_list, label=f"connected_component:{bina_method}"):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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

        #Merge diacritics with their nearest main symbol (vertically aligned)
        #Each diacritic is assigned to only the closest aligned main symbol, not every aligned one
        diacritic_best_match = {}
        for symbol_pos, symbol in enumerate(main_symbols):
            idx, x, y, w, h, area = symbol
            symbol_center_x = x + w // 2
            for diac_pos, diacritic in enumerate(diacritics):
                diac_idx, dx, dy, dw, dh, d_area = diacritic
                diac_center_x = dx + dw // 2
                #Check if diacritic is roughly aligned horizontally
                distance = abs(diac_center_x - symbol_center_x)
                if distance < max(w, dw) / 2:
                    if diac_pos not in diacritic_best_match or distance < diacritic_best_match[diac_pos][1]:
                        diacritic_best_match[diac_pos] = (symbol_pos, distance)

        merged_symbols = []
        for symbol_pos, symbol in enumerate(main_symbols):
            idx, x, y, w, h, area = symbol
            merged_indices = [idx]  #Start with the main symbol

            for diac_pos, (matched_symbol_pos, _) in diacritic_best_match.items():
                if matched_symbol_pos == symbol_pos:
                    merged_indices.append(diacritics[diac_pos][0])

            #Merge bounding boxes
            x_min = min([x] + [stats[i][0] for i in merged_indices])
            y_min = min([y] + [stats[i][1] for i in merged_indices])
            x_max = max([x + w] + [stats[i][0] + stats[i][2] for i in merged_indices])
            y_max = max([y + h] + [stats[i][1] + stats[i][3] for i in merged_indices])
            new_w, new_h = x_max - x_min, y_max - y_min

            merged_symbols.append((merged_indices, x_min, y_min, new_w, new_h))

        #Save each merged symbol
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

            doc_id = os.path.splitext(filename)[0]
            symbol_filename = os.path.join(output_folder, f"symbol_{doc_id}_{global_counter}.png")
            cv2.imwrite(symbol_filename, cleaned_symbol)
            
            #Increment the global counter
            global_counter += 1

            #raw bounding box (for visualization)
            cv2.rectangle(boxes_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #Save visualization with bounding boxes
        boxes_filename = os.path.join(boxes_folder, f"boxes_{filename}")
        cv2.imwrite(boxes_filename, boxes_image)

#This is the character-crop count that drives the next stage's autoencoder batch count and,
#more importantly, the O(n^2) silhouette_score search in hierarchical_silhouette_bucle.py -
#worth watching for: tens of thousands is fine, well past ~50-80K gets slow/memory-heavy.
print(f"[connected_component:{bina_method}] done - extracted {global_counter} character crops "
      f"from {len(file_list)} documents", flush=True)