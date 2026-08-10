import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.utils.binarization_method import resolve_binarization_method
from model.utils.progress import progress
from model.utils.parallel import parallel_map_unordered
from model.utils.cleanup import cleanup_stage_output

#Initial setup

DATASET_PATH = os.path.expanduser('~/Caramba/Dataset/corpus_cipherTypeFinder_Caramba')

#Method forwarded as argv[1] when launched by main_preprocessing.py; falls back to an
#interactive prompt when run standalone.
bina_method = resolve_binarization_method(sys.argv[1] if len(sys.argv) > 1 else None)


image_folder = f"{DATASET_PATH}/preprocessing/connectedComponent/{bina_method}/symbols"
output_folder = f"{DATASET_PATH}/preprocessing/processed"
os.makedirs(output_folder, exist_ok=True)

TARGET_SIZE = (100, 100)


def _read_and_resize(img_path):
    """Read+resize a single character crop. Runs in a worker process - a pure function of its
    path argument (uses the module-level TARGET_SIZE constant rather than a closure, since a
    closure over a local variable can't be pickled for a process pool)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read {img_path}", flush=True)
        return None

    if 0 in img.shape:
        print(f"Warning: Skipping empty image {img_path}", flush=True)
        return None

    return resize_with_white_padding(img, TARGET_SIZE)


def preprocess_images_simple(image_folder, output_folder, target_size=TARGET_SIZE):
    #Get all image paths
    image_paths = list(Path(image_folder).glob("*.png")) + \
                  list(Path(image_folder).glob("*.jpg")) + \
                  list(Path(image_folder).glob("*.jpeg")) + \
                  list(Path(image_folder).glob("*.bmp")) + \
                  list(Path(image_folder).glob("*.tif")) + \
                  list(Path(image_folder).glob("*.tiff"))

    if not image_paths:
        print(f"No images found in {image_folder}")
        return None, None, 0

    n_candidates = len(image_paths)
    print(f"Found {n_candidates} images for preprocessing")

    #Process all images with white padding
    print("Preprocessing images with white padding...")

    #Write directly to disk via a memory-mapped .npy instead of accumulating every
    #image in a Python list and converting to one big array at the end. That old
    #approach held two full copies of the data in RAM momentarily and only touched
    #disk once, right at the finish line, after hours of work -- one write failure
    #(e.g. disk quota) lost everything. Streaming writes as we go means a failure
    #is caught early and the process never needs the whole array resident in RAM.
    #Preallocated at the candidate count (worst case: every image is valid); rare
    #unreadable/empty images just leave unused rows at the end, trimmed below.
    data_path = Path(output_folder) / "preprocessed_data.npy"
    processed_array = np.lib.format.open_memmap(
        str(data_path), mode='w+', dtype=np.uint8, shape=(n_candidates, *target_size)
    )

    image_filenames = []
    valid_count = 0

    #Reading + resizing each crop is independent of every other one and CPU-bound (cv2.resize),
    #so it fans out across a process pool (see model/utils/parallel.py). The memmap write itself
    #stays sequential in this process - interleaving writes to the same memmap from multiple
    #processes isn't safe - but that's cheap next to the decode/resize work, so this still overlaps
    #disk I/O in the main process with CPU work happening in the workers.
    for img_path, resized in progress(parallel_map_unordered(_read_and_resize, image_paths),
                                       total=n_candidates, label="resize_for_clustering"):
        if resized is None:
            continue

        processed_array[valid_count] = resized
        image_filenames.append(img_path.name)
        valid_count += 1

    processed_array.flush()

    if valid_count == 0:
        del processed_array
        data_path.unlink(missing_ok=True)
        print("No images were processed. Please check the input folder path.")
        return None, None, 0

    if valid_count < n_candidates:
        #A handful of source images were skipped (unreadable/empty). The valid rows
        #were written in order starting at index 0, so they're a contiguous prefix
        #of the file; repack into a correctly-shaped array rather than shipping a
        #.npy with stale/zeroed trailing rows or hand-patching the header in place.
        skipped = n_candidates - valid_count
        print(f"Repacking array to drop {skipped} skipped slot(s)...")
        tmp_path = data_path.with_suffix(".tmp.npy")
        trimmed = np.lib.format.open_memmap(
            str(tmp_path), mode='w+', dtype=np.uint8, shape=(valid_count, *target_size)
        )
        trimmed[:] = processed_array[:valid_count]
        trimmed.flush()
        del processed_array, trimmed
        tmp_path.replace(data_path)
    else:
        del processed_array

    print(f"Successfully preprocessed {valid_count} images")
    return data_path, image_filenames, valid_count

def resize_with_white_padding(img, target_size=(100, 100)):
    h, w = img.shape
    target_h, target_w = target_size
    
    #Calculate scaling factor
    scale = min(target_h / h, target_w / w)
    #Clamp to at least 1px: a very thin/elongated crop can otherwise round down to 0,
    #which cv2.resize rejects (inv_scale_x/y > 0 assertion)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    
    #Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    #Calculate padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    
    #Use white padding
    padding_value = 255
    
    #Apply padding (equal padding on both sides when possible)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    #Apply padding with white value
    padded = cv2.copyMakeBorder(
        resized, 
        top, bottom, left, right, 
        cv2.BORDER_CONSTANT, 
        value=padding_value
    )
    
    return padded


if __name__ == "__main__":
    #Process all images with simple preprocessing (padding only)
    data_path, image_filenames, n_processed = preprocess_images_simple(image_folder, output_folder)

    if data_path:
        print(f"\nPreprocessing completed successfully!")
        print(f"Number of processed images: {n_processed}")

        #Report shape/dtype via the memmap header only -- avoids pulling the full
        #array back into RAM just to print a sanity check.
        processed_array = np.load(str(data_path), mmap_mode='r')
        print(f"\nProcessed data shape: {processed_array.shape}")
        print(f"Data type: {processed_array.dtype}")
        del processed_array
        print(f"Preprocessed data saved to: {data_path}")

        #Filenames kept as a plain JSON list alongside the array, instead of pickling
        #{'data': ..., 'filenames': ...} into the .npy -- that forced allow_pickle and
        #meant the array itself could never be memory-mapped back out.
        filenames_path = Path(output_folder) / "preprocessed_filenames.json"
        with open(filenames_path, 'w') as f:
            json.dump(image_filenames, f)
        print(f"Filenames saved to: {filenames_path}")

        #Nothing downstream ever reads the individual character crops again once they're baked
        #into preprocessed_data.npy - reclaim the space now instead of letting every stage's
        #output pile up on disk at once. Only reached once the array and filenames are both
        #safely persisted above, so a mid-run failure never loses the only copy of this data.
        cleanup_stage_output(image_folder, "extracted character crops")

    else:
        print("No images were processed. Please check the input folder path.")