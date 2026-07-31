###
#Launcher for preprocessing process
#Take as input dataset images and output ready-to-use data for the model described into "Unsupervised Feature Learning via Convolutional Autoencoders for
#Cross-Manuscript Comparison in Historical Cryptanalysis" from Alejandra Reinares, Giuseppe De Gregorio and Alicia Fornés
###

import os


def main_prepro():
    #1) Binarization process
    os.system("python3 ./binarize.py")

    #2) Line segmentation process
    os.system("python3 ./line_segmentation.py")

    #3) Cleaning process
    os.system("python3 ./cleaning.py")

    #4) Connected Component process
    os.system("python3 ./connected_component.py")

    #5) Processing for Clustering process
    os.system("python3 ./processing_for_clustering.py")