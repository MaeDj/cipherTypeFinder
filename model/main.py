###
#Main dedicated to the model preprocessing + training
###

from preprocessing import main_preprocessing
from txt_equivalent_builder import txtBuilder
from statistics import equivalent_txt_manuscripts_stats
from text_clusterization import MLP_cipher_classifier
from convolutional_autoEncoder import hierarchical_silhouette_bucle
from utils.binarization_method import resolve_binarization_method

###
#Manuscript detection training
###

#Asked once here; every preprocessing stage that needs it receives this value instead
#of prompting for it itself.
BINARIZATION_METHOD = resolve_binarization_method()

###
#Preprocessing
###

#All files from the corpus are preprocessed and stored into ./corpus folder
main_preprocessing.main_prepro(BINARIZATION_METHOD)


###
# Characters are clusterized
###

hierarchical_silhouette_bucle.main()
###
#TXT equivalent to original manuscript creation
###
txtBuilder.main()

###
#Statistics computation
###
equivalent_txt_manuscripts_stats.main()

###
#Manuscript clusterization and final results
###

model, label_names, encoders=MLP_cipher_classifier.main()

