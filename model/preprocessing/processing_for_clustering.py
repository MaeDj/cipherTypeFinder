import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

#Initial setup

DATASET_PATH = f'../corpus'

binarization_methods = ["otsu","gauss","adaptive","niblack","sauvola","local"]
bina_method = False
while (bina_method not in ["","1","2","3","4","5", "6"]):
	bina_method = input("Select the binarization method that you used: [Default:5] (1:Otsu 2:Gaussian 3:Adaptive 4:Niblack 5:Sauvola 6:Local) \n")
if bina_method == "":
	bina_method = "5"

bina_method = binarization_methods[int(bina_method)-1]


image_folder = f"{DATASET_PATH}/preprocessing/connectedComponent/{bina_method}/symbols"
output_folder = f"{DATASET_PATH}/preprocessing/processed"
os.makedirs(output_folder, exist_ok=True)


def preprocess_images_simple(image_folder, target_size=(100, 100)):
    #Get all image paths
    image_paths = list(Path(image_folder).glob("*.png")) + \
                  list(Path(image_folder).glob("*.jpg")) + \
                  list(Path(image_folder).glob("*.jpeg")) + \
                  list(Path(image_folder).glob("*.bmp")) + \
                  list(Path(image_folder).glob("*.tif")) + \
                  list(Path(image_folder).glob("*.tiff"))
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return None, None
    
    print(f"Found {len(image_paths)} images for preprocessing")
    
    #Process all images with white padding
    print("Preprocessing images with white padding...")
    processed_images = []
    image_filenames = []
    
    for img_path in image_paths:
        #Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        #Resize with preserved aspect ratio and white padding
        processed_img = resize_with_white_padding(img, target_size)
        
        processed_images.append(processed_img)
        image_filenames.append(img_path.name)
        
        #Save preprocessed images
        save_preprocessed = True  #Set to False if you don't want to save
        if save_preprocessed:
            preprocessed_folder = Path(output_folder) / "images"
            preprocessed_folder.mkdir(exist_ok=True)
            save_path = preprocessed_folder / img_path.name
            cv2.imwrite(str(save_path), processed_img)
    
    print(f"Successfully preprocessed {len(processed_images)} images")
    return processed_images, image_filenames

def resize_with_white_padding(img, target_size=(100, 100)):
    h, w = img.shape
    target_h, target_w = target_size
    
    #Calculate scaling factor
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
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


#Process all images with simple preprocessing (padding only)
processed_images, image_filenames = preprocess_images_simple(image_folder)

if processed_images:
    print(f"\nPreprocessing completed successfully!")
    print(f"Number of processed images: {len(processed_images)}")
    
    #Convert processed images to numpy array for clustering
    processed_array = np.array(processed_images)
    print(f"\nProcessed data shape: {processed_array.shape}")
    print(f"Data type: {processed_array.dtype}")
    print(f"Data range: [{processed_array.min()}, {processed_array.max()}]")
    
    #Save processed data for clustering
    data_path = Path(output_folder) / "preprocessed_data.npy"
    np.save(str(data_path), {
        'data': processed_array,
        'filenames': image_filenames
    })
    print(f"Preprocessed data saved to: {data_path}")
    
else:
    print("No images were processed. Please check the input folder path.")