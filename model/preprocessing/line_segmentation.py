import os
import sys
from pathlib import Path
from natsort import natsorted
import cv2
import peakutils
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.utils.remote_listdir import listdir as remote_listdir
from model.utils.binarization_method import resolve_binarization_method

#Initial setup

DATASET_PATH = os.path.expanduser('~/Caramba/Dataset/corpus_cipherTypeFinder_Caramba')

#Method forwarded as argv[1] when launched by main_preprocessing.py; falls back to an
#interactive prompt when run standalone.
bina_method = resolve_binarization_method(sys.argv[1] if len(sys.argv) > 1 else None)

input_folder =f"{DATASET_PATH}/preprocessing/binarized/{bina_method}"
output_folder =f"{DATASET_PATH}/preprocessing/line_segmented/{bina_method}"
os.makedirs(output_folder, exist_ok=True)

def projectionLines(img, minDistLineSeg, thresLineSeg):  #Calculates horizontal projection to find the center of text lines
    (rows, cols) = img.shape
    #Calculate horizontal projection
    h_projection = np.array([x / 255 / cols for x in img.sum(axis=1)])
    h_projection = abs(1 - h_projection)
    
    #Identify peaks in the projection (representing lines)
    indicesObj = peakutils.indexes(h_projection, thres=thresLineSeg, min_dist=minDistLineSeg)
    
    #Calculate average distance between lines
    aux_peaks = [0] + list(indicesObj) + [rows]
    line_mean = 0
    for i in range(len(aux_peaks) - 1):
        line_mean += (aux_peaks[i+1] - aux_peaks[i])
    line_mean /= len(aux_peaks) - 1

    return indicesObj, line_mean

def connectedComponentsToLines(image, peaks):  #Groups connected components based on which peak (line) they are closest to
    #Inside connectedComponentsToLines
    _, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    output = cv2.connectedComponentsWithStats(binary_img, 4, cv2.CV_32S)
    print(f"Found {output[0]} individual components in the image.")
    symbols = []
    
    for indCC in range(1, output[0]):
        stat = output[2][indCC]
        centroid = output[3][indCC]
        
        #Determine which line peak this component belongs to
        minDist = 9999
        minInd = -1
        
        #Check if component overlaps a peak
        touch = False
        for indPeaks in range(0, len(peaks)):
            if stat[cv2.CC_STAT_TOP] <= peaks[indPeaks] <= stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT]:
                minInd = indPeaks
                touch = True
                break
        
        #If no overlap, assign to the nearest peak
        if not touch:
            for indPeaks in range(0, len(peaks)):
                if abs(centroid[1] - peaks[indPeaks]) < minDist:
                    minDist = abs(centroid[1] - peaks[indPeaks])
                    minInd = indPeaks
        
        #Extract the component mask
        left, top, w, h = stat[0], stat[1], stat[2], stat[3]
        crop = np.array(output[1][top:top+h, left:left+w])
        crop[crop != indCC] = 0
        crop[crop == indCC] = 1
        
        symbols.append([crop.astype(bool), centroid, minInd, stat])
        
    return symbols

def saveSegmentedLines(image, symbols, peaks, output_folder, filename, minDistLineSeg):  #Reconstructs full lines from grouped components and saves them
    lines_count = 0
    for indPeak in range(0, len(peaks)):
        mask = np.zeros(image.shape, dtype=bool)
        line_has_content = False
        
        for sym in symbols:
            if sym[2] == indPeak:
                stat = sym[3]
                left, top, w, h = stat[0], stat[1], stat[2], stat[3]
                mask[top:top+h, left:left+w] |= sym[0]
                line_has_content = True
        
        if line_has_content:
            #Crop the line mask to its content height
            active_rows = np.where(mask.max(axis=1) > 0)[0]
            if len(active_rows) > 0:
                firstRow, lastRow = active_rows[0], active_rows[-1]
                line_img = mask[firstRow:lastRow+1, :]
                
                #Check if line height meets minimum requirement
                if line_img.shape[0] >= minDistLineSeg:
                    #Invert back (text as black on white) and save
                    final_line = (True ^ line_img) * 255
                    out_name = filename.rsplit('.', 1)[0] + f'_line_{indPeak}.png'
                    cv2.imwrite(os.path.join(output_folder, out_name), final_line.astype(np.uint8))
                    lines_count += 1
    return lines_count

#Parameters (Adjust based on specific cipher images)
minDistLineSeg = 50 
thresLineSeg = 0.2

for filename in natsorted(remote_listdir(input_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        #Load Image
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path, 0)
        
        if image is None:
            continue

        #Get line peaks via horizontal projection
        peaks, line_avg_dist = projectionLines(image, minDistLineSeg, thresLineSeg)
        
        #Get connected components and assign them to line indices
        symbols_grouped = connectedComponentsToLines(image, peaks)
        
        #Save the reconstructed lines
        num_saved = saveSegmentedLines(image, symbols_grouped, peaks, output_folder, filename, minDistLineSeg)