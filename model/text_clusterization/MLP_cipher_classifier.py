###
#Multi-label MLP classifier: predicts which cipher type(s) a manuscript uses from the
#character-repartition statistics computed by ../statistics/equivalent_txt_manuscripts_stats.py
#Reference: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
###
import re
import numpy as np
import pandas as pd
from numpy.f2py.symbolic import as_ge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder
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


#Split a comma-separated metadata field (eg cipher_types "1,4" or symbol_sets "1") into a list of
#stripped tokens. Missing/empty values are facultative and simply yield no token instead of raising
def split_codes(raw_value):
    if not isinstance(raw_value, str) or not raw_value.strip():
        return []
    return [token.strip() for token in raw_value.split(',') if token.strip()]


#cipher_types is stored as a comma-separated string (eg "1,4") since a single manuscript can combine
#several cipher types, which makes this a multi-label problem
def parse_cipher_types(raw_value):
    return [int(token) for token in split_codes(raw_value) if token.isdigit() and int(token) in CIPHER_TYPE_LABELS]


#plaintext_lang may list several languages informally (eg "French? Portuguese?"), so only the
#alphabetic tokens are kept, lowercased, regardless of punctuation/separators used
def parse_plaintext_lang(raw_value):
    if not isinstance(raw_value, str):
        return []
    return sorted(set(token.lower() for token in re.findall(r'[A-Za-z]+', raw_value)))


#All metadata fields below are facultative: a document missing one of them must still be encoded
#(not dropped, not erroring), which is why every encoder here is fit to tolerate missing/unseen input
#(OneHotEncoder(handle_unknown='ignore'), MultiLabelBinarizer on a possibly-empty token list, median
#imputation for start_year).
#Build the feature matrix (character-repartition statistics plus optional origin/start_year/
#plaintext_lang/symbol_sets metadata) and the multi-label target matrix (one binary column per
#cipher type, ordered like CIPHER_TYPE_LABELS) from the stats .csv. Also returns the fitted encoders
#so a new document can later be encoded the same way (see build_feature_row)
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    df['cipher_type_codes'] = df['cipher_types'].apply(parse_cipher_types)
    #documents with no recognized cipher type carry no usable label, drop them
    #Not supposed to exist, excluded in fetch_data.py
    df = df[df['cipher_type_codes'].map(len) > 0]

    numeric_columns = [col for col in df.columns
                        if col in ('total_symbols', 'alphabet_size', 'coincidence_index', 'start_year')
                        or col.startswith('freq_')] #for each symbol into the whole dataset
    numeric_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    numeric_medians = numeric_df.median()
    numeric_features = numeric_df.fillna(numeric_medians).to_numpy()

    origin_encoder = OneHotEncoder(handle_unknown='ignore')
    origin_features = origin_encoder.fit_transform(df[['origin']].fillna('unknown').astype(str)).toarray()

    lang_binarizer = MultiLabelBinarizer()
    lang_features = lang_binarizer.fit_transform(df['plaintext_lang'].apply(parse_plaintext_lang))

    symbol_set_binarizer = MultiLabelBinarizer()
    symbol_set_features = symbol_set_binarizer.fit_transform(df['symbol_sets'].apply(split_codes))

    x = np.hstack([numeric_features, origin_features, lang_features, symbol_set_features])

    mlb = MultiLabelBinarizer(classes=sorted(CIPHER_TYPE_LABELS.keys()))
    y = mlb.fit_transform(df['cipher_type_codes'])
    label_names = [CIPHER_TYPE_LABELS[code] for code in mlb.classes_]

    encoders = {
        'numeric_columns': numeric_columns,
        'numeric_medians': numeric_medians,
        'origin_encoder': origin_encoder,
        'lang_binarizer': lang_binarizer,
        'symbol_set_binarizer': symbol_set_binarizer,
    }

    return x, y, label_names, encoders


#Train an MLP on the statistics and report its multi-label performance on a held-out split
def main():
    x, y, label_names, encoders = load_dataset(INPUT_CSV)

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

    return model, label_names, encoders


#Build a single document's feature vector the same way load_dataset does, using its fitted encoders.
#stats holds the character-repartition values (total_symbols, alphabet_size, coincidence_index,
#start_year, freq_*); origin/plaintext_lang/symbol_sets are all optional and default to their
#facultative encoding (median, 'unknown' origin, no language/symbol-set token) when None
def build_feature_row(stats, encoders, origin=None, plaintext_lang=None, symbol_sets=None):
    numeric_medians = encoders['numeric_medians']
    numeric_values = [
        stats[column] if stats.get(column) is not None else numeric_medians[column]
        for column in encoders['numeric_columns']
    ]

    origin_value = pd.DataFrame([[origin if origin is not None else 'unknown']], columns=['origin'])
    origin_features = encoders['origin_encoder'].transform(origin_value).toarray()[0]

    lang_features = encoders['lang_binarizer'].transform([parse_plaintext_lang(plaintext_lang)])[0]
    symbol_set_features = encoders['symbol_set_binarizer'].transform([split_codes(symbol_sets)])[0]

    return np.concatenate([numeric_values, origin_features, lang_features, symbol_set_features])


#Predict the cipher type(s) of a single document from its statistics row (same feature order
#as load_dataset/build_feature_row), returning every label whose predicted probability clears the threshold
def predict_cipher_types(model, label_names, feature_row, threshold=0.5):
    probabilities = model.predict_proba([feature_row])[0]
    return [name for name, probability in zip(label_names, probabilities) if probability >= threshold]


main()