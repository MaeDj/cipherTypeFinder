###
#Main dedicated to the model preprocessing + training
###

from preprocessing import main_preprocessing
from txt_equivalent_builder import txtBuilder
from statistics import equivalent_txt_manuscripts_stats
from text_clusterization import MLP_cipher_classifier
from convolutional_autoEncoder import hierarchical_silhouette_bucle
from utils.binarization_method import resolve_binarization_method
from utils.progress import stage

###
#Manuscript detection training
###

#Asked once here; every preprocessing stage that needs it receives this value instead
#of prompting for it itself.
BINARIZATION_METHOD = resolve_binarization_method()

with stage("FULL PIPELINE"):

    ###
    #Preprocessing
    ###

    #All files from the corpus are preprocessed and stored into ./corpus folder
    with stage("1/5 Preprocessing (binarize -> line segment -> clean -> connected components -> resize)"):
        main_preprocessing.main_prepro(BINARIZATION_METHOD)


    ###
    # Characters are clusterized
    ###

    with stage("2/5 Character clustering (autoencoder + hierarchical silhouette)"):
        hierarchical_silhouette_bucle.main()

    ###
    #TXT equivalent to original manuscript creation
    ###
    with stage("3/5 Computable .txt equivalent creation"):
        txtBuilder.main()

    ###
    #Statistics computation
    ###
    with stage("4/5 Manuscript statistics computation"):
        equivalent_txt_manuscripts_stats.main()

    ###
    #Manuscript clusterization and final results
    ###

    with stage("5/5 Cipher-type MLP training"):
        model, label_names, encoders = MLP_cipher_classifier.main()

