###
#This algorithm's aim is to get from .txt files tabular informations about characters repartition into the ciphered text
#
###
import os
import re
import csv
import json

ORIGINAL_DATASET_METADATA_PATH='/media/mae/PHILIPS UFD/corpus_cipherTypeFinder_Caramba/original_corpus/metadata'
#dataset folder root
DATASET_PATH = '../corpus'
#input directory, written by ../txt_equivalent_builder/txtBuilder.py
INPUT_DIRECTORY = f"{DATASET_PATH}/computable/text"
#output directory for the statistics .csv
OUTPUT_DIRECTORY = f"{DATASET_PATH}/computable/statistics"
OUTPUT_CSV = f"{OUTPUT_DIRECTORY}/manuscripts_stats.csv"

#metadata columns written by ../corpus_builder/fetch_data.py, identical across every document
#(origin_region/origin_city are merged into a single 'origin' column, see read_metadata)
METADATA_FIELDS = ['origin', 'start_year', 'cipher_types']


#Read a computable .txt equivalent and split it into its ordered list of symbols
def read_symbols(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as txt_file:
        content = txt_file.read()
    return content.split()


#Number of distinct symbols used into the document (size of its alphabet)
def alphabet_size(symbols):
    return len(set(symbols))


#Occurrence count and frequency (count/total) of each symbol into the document
def symbol_frequencies(symbols):
    total = len(symbols)
    counts = dict()
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1

    frequencies = dict()
    for symbol, count in counts.items():
        frequencies[symbol] = count / total if total > 0 else 0

    return counts, frequencies


#Index of coincidence: probability that two symbols drawn at random from the document are identical
def coincidence_index(symbols):
    total = len(symbols)
    if total < 2:
        return 0

    counts, _ = symbol_frequencies(symbols)
    numerator = sum(count * (count - 1) for count in counts.values())
    return numerator / (total * (total - 1))


#Read a document's original metadata written by ../corpus_builder/fetch_data.py
#origin_region and origin_city are merged into a single 'origin' field, giving origin_city priority
#and falling back to origin_region (fetch_data.py only ever requires at least one of the two to be set)
def read_metadata(doc):
    metadata_path = os.path.join(ORIGINAL_DATASET_METADATA_PATH, f"{doc}.json")
    if not os.path.isfile(metadata_path):
        return {field: '' for field in METADATA_FIELDS}

    with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
        metadata = json.load(metadata_file)

    return {
        'origin': metadata.get('origin_city') or metadata.get('origin_region') or '',
        'start_year': metadata.get('start_year', ''),
        'cipher_types': metadata.get('cipher_types', ''),
    }


#Find every document id from the .txt files available into the input directory
def doc_id():
    docId_list = []
    for filename in os.listdir(INPUT_DIRECTORY):
        match = re.match(r'(.+)\.txt$', filename.lower())
        if match:
            docId_list.append(match.group(1))
    return docId_list


#Register the original metadata, alphabet size, coincidence index and each symbol's frequency of
#every document into a single .csv
#One row per document, one freq_<symbol> column per symbol found across the whole corpus, so
#alphabet_size/coincidence_index are only ever stored once per document instead of once per symbol row
def register_statistics(dict_doc_symbols):
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    #union of every symbol used across the whole corpus, to give every document the same columns
    corpus_alphabet = sorted(set(symbol for symbols in dict_doc_symbols.values() for symbol in symbols))

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['doc_id'] + METADATA_FIELDS + ['total_symbols', 'alphabet_size', 'coincidence_index'] + [f"freq_{symbol}" for symbol in corpus_alphabet])
#WARNING: to use coincidence index, there need to also use plaintext language metadata.
        for doc, symbols in dict_doc_symbols.items():
            _, frequencies = symbol_frequencies(symbols)
            metadata = read_metadata(doc)

            row = [doc] + [metadata[field] for field in METADATA_FIELDS]
            row += [len(symbols), alphabet_size(symbols), coincidence_index(symbols)]
            row += [frequencies.get(symbol, 0) for symbol in corpus_alphabet]
            writer.writerow(row)



def main():

    #find every document id available as a computable .txt equivalent
    docId_list = doc_id()

    #split each document into its ordered list of symbols
    dict_doc_symbols = dict()
    for doc in docId_list:
        txt_path = os.path.join(INPUT_DIRECTORY, f"{doc}.txt")
        dict_doc_symbols[doc] = read_symbols(txt_path)

    #compute and register alphabet size, coincidence index and symbol frequencies for every document at once
    register_statistics(dict_doc_symbols)

    print(f"Statistics computed for {len(docId_list)} document(s), written to {OUTPUT_CSV}")

    #directly written into ../corpus/computable/statistics/manuscripts_stats.csv

main()