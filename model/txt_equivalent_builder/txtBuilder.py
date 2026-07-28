###
#This script is dedicated to the conversion of handwritten manuscript to computable txt.
#Each character of each document of the corpus were clusterized.
#The goal is to reconstruct each document by finding each associated character into the right order and write into the txt the corresponding translation using the associate cluster's computable character
###
LATENT_DIM = 64

#dataset folder root
DATASET_PATH=f'../corpus'


#output directory of ../Convolutional_autoEncoder/hierarchical_silhouette_bucle.py
INPUT_DIRECTORY=f"{DATASET_PATH}/clustering/hierarchical/clusters_{str(LATENT_DIM)}_silhouette_results"
OUTPUT_DIRECTORY=f"{DATASET_PATH}/computable/"


