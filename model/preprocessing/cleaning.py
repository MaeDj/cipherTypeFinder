import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.utils.remote_listdir import listdir as remote_listdir
from model.utils.binarization_method import resolve_binarization_method


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

for image in remote_listdir(input_folder):
    if image.lower().endswith(('.png', '.jpg', '.jpeg')):
        #Load and preprocess the image
        binary_image = cv2.imread(os.path.join(input_folder, image), cv2.IMREAD_GRAYSCALE)
        
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
            continue  #Skip if no foreground pixels
            
        X = x_coords.reshape(-1, 1)
        y = y_coords.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        margin = 5
        middle_zone_upper = (y_pred - margin).flatten()
        middle_zone_lower = (y_pred + margin).flatten()
        
        #Create a map of middle zone for each x coordinate
        middle_zone_map = {}
        for x, upper, lower in zip(x_coords, middle_zone_upper, middle_zone_lower):
            middle_zone_map[x] = (upper, lower)
        
        #Connected component analysis
        num_labels, labels, stats, centroids = connected_components_with_stats(255 - binary_image, connectivity=8)
        
        #Get image dimensions
        height, width = binary_image.shape
        
        #Create a mask for components to remove (initially all False)
        remove_mask = np.zeros_like(binary_image, dtype=bool)
        
        #Iterate over connected components (skip background label 0)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            #Check if component touches top or bottom border
            touches_top = y == 0
            touches_bottom = y + h >= height - 1
            
            #Check if component reaches middle zone
            reaches_middle = False
            
            #Get all pixels in the component
            component_mask = (labels == i).astype(np.uint8)
            component_pixels = np.argwhere(component_mask)
            
            for py, px in component_pixels:
                if px in middle_zone_map:
                    upper, lower = middle_zone_map[px]
                    if upper <= py <= lower:
                        reaches_middle = True
                        break
            
            #If component touches border but doesn't reach middle zone, mark for removal
            if ((touches_top or touches_bottom) and not reaches_middle) or area<=MIN_AREA:
                remove_mask = np.logical_or(remove_mask, (labels == i))

        #Clean the original image (set removed components to white)
        cleaned_image = binary_image.copy()
        cleaned_image[remove_mask] = 255  #Set removed components to white
        
        #Save the cleaned image
        cleaned_image_path = os.path.join(cleaned_output_folder, image)
        cv2.imwrite(cleaned_image_path, cleaned_image)
