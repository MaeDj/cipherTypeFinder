###
#Multi-label MLP classifier: predicts which cipher type(s) a manuscript uses from the
#character-repartition statistics computed by ../statistics/equivalent_txt_manuscripts_stats.py
#Reference: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
###
import pandas as pd
from numpy.f2py.symbolic import as_ge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, hamming_loss, jaccard_score, accuracy_score

###
#BODY
###

INPUT_PATH = '../corpus/computable/statistics'
INPUT_CSV = f'{INPUT_PATH}/manuscripts_stats.csv'

#cipher_types codes come from ../corpus_builder/fetch_data.py (code 6 is already excluded at fetch time)
CIPHER_TYPE_LABELS = {
    1: 'monoalphabetic',
    2: 'homophonic',
    3: 'machine',
    4: 'polyalphabetic',
    5: 'nomenclature',
    7: 'polyphonic',
}


#cipher_types is stored as a comma-separated string (eg "1,4") since a single manuscript can combine
#several cipher types, which makes this a multi-label problem
def parse_cipher_types(raw_value):
    if not isinstance(raw_value, str) or not raw_value.strip():
        return []
    codes = []
    for token in raw_value.split(','):
        token = token.strip()
        if token.isdigit() and int(token) in CIPHER_TYPE_LABELS:
            codes.append(int(token))
    return codes


#Build the feature matrix (character-repartition statistics) and the multi-label target matrix
#(one binary column per cipher type, ordered like CIPHER_TYPE_LABELS) from the stats .csv
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    df['cipher_type_codes'] = df['cipher_types'].apply(parse_cipher_types)
    #documents with no recognized cipher type carry no usable label, drop them
    #Not supposed to exist, excluded in fetch_data.py
    df = df[df['cipher_type_codes'].map(len) > 0]

    feature_columns = [col for col in df.columns
                        if col in ('total_symbols', 'alphabet_size', 'coincidence_index')
                        or col.startswith('freq_')] #for each symbol into the whole dataset
    x = df[feature_columns].fillna(0).to_numpy()

    mlb = MultiLabelBinarizer(classes=sorted(CIPHER_TYPE_LABELS.keys()))
    y = mlb.fit_transform(df['cipher_type_codes'])
    label_names = [CIPHER_TYPE_LABELS[code] for code in mlb.classes_]

    return x, y, label_names


#Train an MLP on the statistics and report its multi-label performance on a held-out split
def main():
    x, y, label_names = load_dataset(INPUT_CSV)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #MLPClassifier supports multi-label classification natively: given a 2D binary target it puts one
    #sigmoid unit per class on the output layer instead of a single softmax (see linked sklearn doc)
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                      max_iter=500, early_stopping=True, random_state=42)
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
    print(f"Exact match accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Hamming loss: {hamming_loss(y_test, y_pred):.3f}")
    print(f"Jaccard score (samples): {jaccard_score(y_test, y_pred, average='samples', zero_division=0):.3f}")

    return model, label_names


#TODO Upgrade: create at the beginning of the pipeline a writer/reader system to allow the user to add informations about his document
#- date v
#- origin v
#- plain text language -> for Coincidence index
#- list of character: numerical, sign, alchemy

#Predict the cipher type(s) of a single document from its statistics row (same feature_columns order
#as load_dataset), returning every label whose predicted probability clears the threshold
def predict_cipher_types(model, label_names, feature_row, threshold=0.5):
    probabilities = model.predict_proba([feature_row])[0]
    return [name for name, probability in zip(label_names, probabilities) if probability >= threshold]


main()