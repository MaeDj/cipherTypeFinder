import cv2
import os
import random
from skimage.filters import threshold_niblack,threshold_sauvola
import numpy as np
from natsort import natsorted

random.seed(0)

#Initial setup
DATASET_PATH = f'/media/mae/PHILIPS UFD/corpus_cipherTypeFinder_Caramba'


binarization_methods = ["otsu","gauss","adaptive","niblack","sauvola","local"]
METHOD = False
while (METHOD not in ["","1","2","3","4","5", "6"]):
	METHOD = input("Select the binarization method that you want to apply: [Default:5] (1:Otsu 2:Gaussian 3:Adaptive 4:Niblack 5:Sauvola 6:Local) \n")
if METHOD == "":
	METHOD = "5"

METHOD = binarization_methods[int(METHOD)-1]

input_folder = f'{DATASET_PATH}/original_corpus/img'
output_folder =f"../corpus/preprocessing/binarized/{METHOD}"

def binarize(im,method):
	if "otsu" in method:
		bina = binarize_otsu(im)
	elif "gauss" in method:
		bina = binarize_otsu_gaussian(im)
	elif "adaptive" in method:
		bina = binarize_adaptiveThreshold(im)
	elif "niblack" in method:
		bina = binarize_niblack(im)
	elif "sauvola" in method:
		bina = binarize_sauvola(im)
	elif "local" in method:
		bina = binarize_localMinimumMaximum(im)
	return bina

def binarize_otsu(im):
	_,bina = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return bina

def binarize_otsu_gaussian(im):
	blur = cv2.GaussianBlur(im,(5,5),0)
	_,bina = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return bina

def binarize_adaptiveThreshold(im):
	bina = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 21)
	return bina

def binarize_niblack(im):
	window_size = 125
	thresh_niblack = threshold_niblack(im, window_size=window_size, k=0.8)
	bina = im > thresh_niblack
	bina = np.uint8(bina*255)
	return bina

def binarize_sauvola(im):
	window_size = 125
	thresh_sauvola = threshold_sauvola(im, window_size=window_size)
	bina = im > thresh_sauvola
	bina = np.uint8(bina*255)
	return bina

def binarize_localMinimumMaximum(im):
	kernel_erode_h = np.ones((1,15),np.uint8)

	gfn = [
		lambda x: np.roll(x, -1, axis=0), 
		lambda x: np.roll(np.roll(x, 1, axis=1), -1, axis=0),
		lambda x: np.roll(x, 1, axis=1),
		lambda x: np.roll(np.roll(x, 1, axis=1), 1, axis=0),
		lambda x: np.roll(x, 1, axis=0),
		lambda x: np.roll(np.roll(x, -1, axis=1), 1, axis=0),
		lambda x: np.roll(x, -1, axis=1),
		lambda x: np.roll(np.roll(x, -1, axis=1), -1, axis=0)
		]

	g = im
	I = g.astype(np.float64)

	cimg = localminmax(I, gfn)
	_, ocimg = cv2.threshold(rescale(cimg).astype(g.dtype), 0, 1, cv2.THRESH_OTSU)

	E = ocimg.astype(np.float64)

	N_e = numnb(ocimg, gfn)
	nbmask = N_e>0

	E_mean = np.zeros(I.shape, dtype=np.float64)
	for fn in gfn:
		E_mean += fn(I)*fn(E)

	E_mean[nbmask] /= N_e[nbmask]

	E_var = np.zeros(I.shape, dtype=np.float64)
	for fn in gfn:
		tmp = (fn(I)-E_mean)*fn(E)
		E_var += tmp*tmp

	E_var[nbmask] /= N_e[nbmask]
	E_std = np.sqrt(E_var)#*.7
	
	R = np.ones(I.shape)*255
	R[(I<=E_mean+E_std)&(N_e>=4)] = 0
	return R

def localminmax(img, fns):
	mi = img.astype(np.float64)
	ma = img.astype(np.float64)
	for i in range(len(fns)):
		rolled = fns[i](img)
		mi = np.minimum(mi, rolled)
		ma = np.maximum(ma, rolled)
	result = (ma-mi)/(mi+ma+1e-16)
	return result

def numnb(bi, fns):
	nb = bi.astype(np.float64)
	i = np.zeros(bi.shape, nb.dtype)
	i[bi==bi.max()] = 1
	i[bi==bi.min()] = 0
	for fn in fns:
		nb += fn(i)
	return nb

def rescale(r,maxvalue=255):
	mi = r.min()
	return maxvalue*(r-mi)/(r.max()-mi)

def start_binarization(method, input_path, output_path):
	for filename in natsorted(os.listdir(input_path), key = lambda x: x.lower()):
		im = cv2.imread(os.path.join(input_path,filename),0)
		bina = binarize(im,method,filename,output_path)
		cv2.imwrite(os.path.join(output_path,filename),bina)



os.makedirs(output_folder, exist_ok=True)

images_list = os.listdir(input_folder)

for image in images_list:
    image_path = os.path.join(input_folder, image)

    #Load image in grayscale
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	
    binary_image = binarize(gray_image, METHOD)

    #Save or display the binarized image
    cv2.imwrite(f'{output_folder}/{image}', binary_image)

