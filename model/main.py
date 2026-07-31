###
#Main dedicated to the model preprocessing + training
###

from preprocessing import main_preprocessing
from txt_equivalent_builder import txtBuilder
from statistics import equivalent_txt_manuscripts_stats
from text_clusterization import MLP_cipher_classifier

###
#Manuscript detection training
###

###
#Preprocessing
###

#All files from the corpus are preprocessed and stored into ./corpus folder
main_preprocessing.main_prepro()

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

