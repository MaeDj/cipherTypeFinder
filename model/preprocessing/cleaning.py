import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
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


input_folder = f"{DATASET_PATH}/preprocessing/line_segmented/{bina_method}"
cleaned_output_folder = f"{DATASET_PATH}/preprocessing/cleaned/{bina_method}"
os.makedirs(cleaned_output_folder, exist_ok=True)

#Minimum area to keep a connected component (adjust as needed)
MIN_AREA = 40

#Was a hand-rolled pure-Python pixel-by-pixel flood fill (visited every pixel individually via a
#Python-level stack) - correct, but easily the single biggest contributor to this stage's runtime
#on full-size manuscript scans. cv2.connectedComponentsWithStats is the same union-find algorithm
#running in C, and returns identically-shaped output: stats rows are [x, y, w, h, area] with label 0
#reserved for background, matching what every call site here already expects.
def connected_components_with_stats(binary_image, connectivity=4):
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be either 4 or 8.")
    return cv2.connectedComponentsWithStats(binary_image, connectivity, cv2.CV_32S)

def _process_one(image):
    """Clean a single line image. Runs in a worker process - relies only on module-level
    globals (input_folder/cleaned_output_folder/MIN_AREA) already set before the pool starts,
    which the default fork start method on Linux hands to every worker for free."""
    if not image.lower().endswith(('.png', '.jpg', '.jpeg')):
        return False

    #Load and preprocess the image
    binary_image = cv2.imread(os.path.join(input_folder, image), cv2.IMREAD_GRAYSCALE)
    if binary_image is None:
        #A single unreadable/corrupt file must not abort the whole cleaning pass.
        print(f"Warning: Could not read {image}, skipping", flush=True)
        return False

    #Invert the image for processing (text becomes white)
    inverted_image = 255 - binary_image

    #Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)
    processed_image = cv2.medianBlur(processed_image, 3)

    #Invert back
    binary_image = 255 - processed_image

    #Extract middle zone
    y_coords, x_coords = np.where(binary_image == 0)

    if len(x_coords) == 0 or len(y_coords) == 0:
        return False  #Skip if no foreground pixels

    X = x_coords.reshape(-1, 1)
    y = y_coords.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    margin = 5

    #Connected component analysis
    num_labels, labels, stats, centroids = connected_components_with_stats(255 - binary_image, connectivity=8)

    #Get image dimensions
    height, width = binary_image.shape

    #Create a mask for components to remove (initially all False)
    remove_mask = np.zeros_like(binary_image, dtype=bool)

    #Iterate over connected components (skip background label 0)
    for i in range(1, num_labels):
        x, y_i, w, h, area = stats[i]

        #Check if component touches top or bottom border
        touches_top = y_i == 0
        touches_bottom = y_i + h >= height - 1

        #Reaches-middle-zone only ever matters for components that touch a border (see the
        #condition below) - was previously computed for every component via a per-pixel Python
        #loop against a dict built from every foreground pixel on the page, which redid work
        #proportional to the whole page's foreground pixel count for components that didn't even
        #need the check. Skipping interior components and vectorizing the rest with numpy over
        #just that component's own pixels turns this into the cheap path it should always have been.
        reaches_middle = False
        if touches_top or touches_bottom:
            comp_ys, comp_xs = np.where(labels == i)
            comp_y_pred = model.predict(comp_xs.reshape(-1, 1)).flatten()
            reaches_middle = bool(np.any((comp_ys >= comp_y_pred - margin) & (comp_ys <= comp_y_pred + margin)))

        #If component touches border but doesn't reach middle zone, mark for removal
        if ((touches_top or touches_bottom) and not reaches_middle) or area <= MIN_AREA:
            remove_mask |= (labels == i)

    #Clean the original image (set removed components to white)
    cleaned_image = binary_image.copy()
    cleaned_image[remove_mask] = 255  #Set removed components to white

    #Save the cleaned image
    cleaned_image_path = os.path.join(cleaned_output_folder, image)
    cv2.imwrite(cleaned_image_path, cleaned_image)
    return True


if __name__ == "__main__":
    file_list = remote_listdir(input_folder)
    print(f"[cleaning:{bina_method}] {len(file_list)} files to scan", flush=True)

    #Every line image is cleaned independently of every other one, so this fans out across a
    #process pool instead of one core handling the whole corpus alone (see model/utils/parallel.py).
    for _image, _ok in progress(parallel_map_unordered(_process_one, file_list),
                                 total=len(file_list), label=f"cleaning:{bina_method}"):
        pass

    #Nothing downstream ever reads the line-segmented images again once they've been cleaned -
    #reclaim the space now instead of letting every stage's output pile up on disk at once.
    cleanup_stage_output(input_folder, "line-segmented images")
