###
#This script is dedicated to the conversion of handwritten manuscript to computable txt.
#Each character of each document of the corpus were clusterized.
#The goal is to reconstruct each document by finding each associated character into the right order and write into the txt the corresponding translation using the associate cluster's computable character
###
import os
import re
import csv


LATENT_DIM = 64

#dataset folder root
DATASET_PATH=f'../corpus'

binarization_methods = ["otsu","gauss","adaptive","niblack","sauvola","local"]
METHOD = False
while (METHOD not in ["","1","2","3","4","5", "6"]):
	METHOD = input("Select the binarization method name used into the data folder you want to explore \n")
if METHOD == "":
	METHOD = "5"

METHOD = binarization_methods[int(METHOD)-1]
#output directory of ../Convolutional_autoEncoder/hierarchical_silhouette_bucle.py
INPUT_DIRECTORY=f"{DATASET_PATH}/clustering/hierarchical/clusters_{str(LATENT_DIM)}_silhouette_results/Merged"
OUTPUT_DIRECTORY=f"{DATASET_PATH}/computable"
ORIGINAL_DIRECTORY=f"../corpus/preprocessing/binarized/{METHOD}"


#Finding all document id

def doc_id():
    docId_list=[]
    for filename in os.listdir(ORIGINAL_DIRECTORY):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            match = re.match(r'(\d+)\.(jpg|jpeg|png)$', filename.lower())
            if match:
                docId_list.append(match.group(1))
    return(docId_list)


#Link each character to his document and provide linked cluster information on position into the document
def organizing_char_list(docId_list):
    dict_doc_charList=dict()
    for doc in docId_list:
        dict_char_info = dict()
        for cluster in os.listdir(os.path.join(INPUT_DIRECTORY,'Accepted')):
            cluster_match = re.match(r'^Cluster_(.+)_Size(\d+)$', cluster)
            if cluster_match:
                cluster_label = cluster_match.group(1)
            else:
                cluster_label = 'Unknown'
                print("Verify cluster folder name")
                continue
            cluster_path = os.path.join(INPUT_DIRECTORY,'Accepted',cluster)
            for char in os.listdir(cluster_path):
                match = re.match(r'^symbol_(.+)_(\d+)\.(jpg|jpeg|png)$', char)
                if match:
                    filename, global_counter = match.group(1), int(match.group(2))
                    if filename==doc:
                        dict_char_info[global_counter]={'cluster':cluster_label,'img_name':char}

                else:
                    print("Verify file name")
                    continue
        dict_doc_charList[doc]=dict_char_info

    return(dict_doc_charList)



#Sort each character depending on their index into the related document
def char_doc_sorting(dict_doc_charList):
    dict_doc_sorted_charlist=dict()
    for doc, dict_char_info in dict_doc_charList.items():
        sorted_info_list=[]
        for index, infoDict in sorted(dict_char_info.items(), key=lambda item: item[0]):
            sorted_info_list.append(infoDict)
        dict_doc_sorted_charlist[doc]=sorted_info_list

    return dict_doc_sorted_charlist


#Create .txt documents from each sorted dictionary
def equivalent_txt_doc_creation(dict_doc_sorted_charlist):
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, 'text'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, 'csv'), exist_ok=True)
    for doc, list_char_info in dict_doc_sorted_charlist.items():
        cluster_labels = [char_info['cluster'] for char_info in list_char_info]


        #Text files used to compute statistics about characters repartition
        txt_path = os.path.join(OUTPUT_DIRECTORY, 'text', f"{doc}.txt")
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(' '.join(cluster_labels))

        #CSV files to keep indications about image path
        csv_path = os.path.join(OUTPUT_DIRECTORY,'csv',f"{doc}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['cluster_label', 'img_name'])
            for char_info in list_char_info:
                writer.writerow([char_info['cluster'], char_info['img_name']])





def main():

    #find doc ID
    docId_List=doc_id()

    #link doc to cluster+char
    dict_doc_charList=organizing_char_list(docId_List)

    #sort following the initial characters order
    sorted_dictDoc_charlist=char_doc_sorting(dict_doc_charList)

    #Create recapitulative .csv + equivalent txt
    equivalent_txt_doc_creation(sorted_dictDoc_charlist)

    #directly written into the folder ../corpus/computable/csv and ../corpus/computable/text

main()







