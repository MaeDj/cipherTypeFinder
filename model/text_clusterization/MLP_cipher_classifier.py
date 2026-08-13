###
#Multi-label MLP classifier: predicts which cipher type(s) a manuscript uses from the
#character-repartition statistics computed by ../statistics/equivalent_txt_manuscripts_stats.py
#Reference: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
###
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, hamming_loss, jaccard_score, accuracy_score

###
#BODY
###

#dataset folder root - absolute path on the data host, so it resolves the
#same regardless of the process cwd
DATASET_PATH = os.path.expanduser('~/Caramba/Dataset/corpus_cipherTypeFinder_Caramba')
INPUT_PATH = f'{DATASET_PATH}/computable/statistics'
INPUT_CSV = f'{INPUT_PATH}/manuscripts_stats.csv'

# Trained model + label names + fitted encoders are persisted here as a single bundle so a later
# CLI call can predict without retraining (see _load_or_train_model and the __main__ block below)
MODEL_PATH = f'{DATASET_PATH}/models/mlp_cipher_classifier.joblib'

#cipher_types codes come from ../corpus_builder/fetch_data.py (code 6 is already excluded at fetch time)
CIPHER_TYPE_LABELS = {
    1: 'monoalphabetic',
    2: 'homophonic',
    3: 'machine',
    4: 'polyalphabetic',
    5: 'nomenclature',
    7: 'polyphonic',
}

# Numeric statistics columns, as written by ../statistics/equivalent_txt_manuscripts_stats.py
# (total_symbols, alphabet_size, coincidence_index, start_year, plus one freq_<symbol> per symbol
# in the corpus). Shared between load_dataset and load_document_stats so both slice the same
# columns out of the stats csv the same way.
def numeric_columns_of(df):
    return [col for col in df.columns
            if col in ('total_symbols', 'alphabet_size', 'coincidence_index', 'start_year')
            or col.startswith('freq_')]


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


#All metadata fields below are facultative: a document missing one of them must still be encoded
#(not dropped, not erroring), which is why every encoder here is fit to tolerate missing/unseen input
#(OneHotEncoder(handle_unknown='ignore'), MultiLabelBinarizer on a possibly-empty token list, median
#imputation for start_year).
#Build the feature matrix (character-repartition statistics plus optional origin/start_year/
#symbol_sets metadata; plaintext_lang is deliberately excluded from the stats .csv, see
#../statistics/equivalent_txt_manuscripts_stats.py) and the multi-label target matrix (one binary
#column per cipher type, ordered like CIPHER_TYPE_LABELS) from the stats .csv. Also returns the fitted
#encoders so a new document can later be encoded the same way (see build_feature_row)
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    df['cipher_type_codes'] = df['cipher_types'].apply(parse_cipher_types)
    #documents with no recognized cipher type carry no usable label, drop them
    #Not supposed to exist, excluded in fetch_data.py
    df = df[df['cipher_type_codes'].map(len) > 0]

    numeric_columns = numeric_columns_of(df)
    numeric_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    numeric_medians = numeric_df.median()
    numeric_features = numeric_df.fillna(numeric_medians).to_numpy()

    origin_encoder = OneHotEncoder(handle_unknown='ignore')
    origin_features = origin_encoder.fit_transform(df[['origin']].fillna('unknown').astype(str)).toarray()

    symbol_set_binarizer = MultiLabelBinarizer()
    symbol_set_features = symbol_set_binarizer.fit_transform(df['symbol_sets'].apply(split_codes))

    x = np.hstack([numeric_features, origin_features, symbol_set_features])

    mlb = MultiLabelBinarizer(classes=sorted(CIPHER_TYPE_LABELS.keys()))
    y = mlb.fit_transform(df['cipher_type_codes'])
    label_names = [CIPHER_TYPE_LABELS[code] for code in mlb.classes_]

    encoders = {
        'numeric_columns': numeric_columns,
        'numeric_medians': numeric_medians,
        'origin_encoder': origin_encoder,
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

    #Persisted as a single bundle (model + label_names + fitted encoders all travel together,
    #since predict_cipher_types/build_feature_row need all three) so a later CLI call can predict
    #without retraining (see _load_or_train_model below)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({'model': model, 'label_names': label_names, 'encoders': encoders}, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    return model, label_names, encoders


#Build a single document's feature vector the same way load_dataset does, using its fitted encoders.
#stats holds the character-repartition values (total_symbols, alphabet_size, coincidence_index,
#start_year, freq_*); coincidence_index is the difference between the document's actual index of
#coincidence and the expected index for its stated plaintext_lang(s), and is None/NaN (imputed to the
#median like every other numeric column) when plaintext_lang names no recognized language.
#origin/symbol_sets are both optional and default to their facultative encoding
#('unknown' origin, no symbol-set token) when None
def build_feature_row(stats, encoders, origin=None, symbol_sets=None):
    numeric_medians = encoders['numeric_medians']
    numeric_values = [
        stats[column] if stats.get(column) is not None else numeric_medians[column]
        for column in encoders['numeric_columns']
    ]

    origin_value = pd.DataFrame([[origin if origin is not None else 'unknown']], columns=['origin'])
    origin_features = encoders['origin_encoder'].transform(origin_value).toarray()[0]

    symbol_set_features = encoders['symbol_set_binarizer'].transform([split_codes(symbol_sets)])[0]

    return np.concatenate([numeric_values, origin_features, symbol_set_features])


#Predict the cipher type(s) of a single document from its statistics row (same feature order
#as load_dataset/build_feature_row), returning every label whose predicted probability clears the threshold
def predict_cipher_types(model, label_names, feature_row, threshold=0.5):
    probabilities = model.predict_proba([feature_row])[0]
    return [name for name, probability in zip(label_names, probabilities) if probability >= threshold]


###
#CLI: predict already-computed documents' cipher type(s) without retraining
###


#Look up a single document's row in the stats csv (step 4's output, INPUT_CSV by default) and
#split it into the (stats, origin, symbol_sets) shape build_feature_row expects - the CLI's way of
#turning a bare doc_id into a feature row, since only that csv (not a raw image) is what this
#classifier's features are computed from.
def load_document_stats(doc_id, csv_path=INPUT_CSV):
    df = pd.read_csv(csv_path)
    matches = df[df['doc_id'] == doc_id]
    if matches.empty:
        raise ValueError(f"No document '{doc_id}' found in {csv_path}")
    row = matches.iloc[0]

    stats = {
        column: (pd.to_numeric(row[column], errors='coerce') if pd.notna(row[column]) else None)
        for column in numeric_columns_of(df)
    }
    origin = row['origin'] if pd.notna(row.get('origin')) else None
    symbol_sets = row['symbol_sets'] if pd.notna(row.get('symbol_sets')) else None

    return stats, origin, symbol_sets


#Load the model bundle saved by main() at MODEL_PATH, training one from scratch (which also saves
#it) if none exists yet or --retrain was passed
def _load_or_train_model(retrain=False):
    if not retrain and os.path.exists(MODEL_PATH):
        print(f"Loading trained model from {MODEL_PATH}")
        bundle = joblib.load(MODEL_PATH)
        return bundle['model'], bundle['label_names'], bundle['encoders']
    return main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_id", nargs="*",
                         help="doc_id(s) already present in the stats csv (INPUT_CSV) to predict "
                              "cipher type(s) for. Omit to just (re)train the model and exit.")
    parser.add_argument("--retrain", action="store_true",
                         help="Retrain from INPUT_CSV even if a saved model already exists at "
                              "MODEL_PATH.")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Minimum per-class probability to report a cipher type (default 0.5).")
    args = parser.parse_args()

    #No doc_id given: this call is just meant to (re)train, so always go through main()
    trained_model, trained_label_names, trained_encoders = _load_or_train_model(
        retrain=args.retrain or not args.doc_id
    )

    for doc_id in args.doc_id:
        doc_stats, doc_origin, doc_symbol_sets = load_document_stats(doc_id)
        feature_row = build_feature_row(doc_stats, trained_encoders, origin=doc_origin, symbol_sets=doc_symbol_sets)
        predicted = predict_cipher_types(trained_model, trained_label_names, feature_row, threshold=args.threshold)
        print(f"{doc_id}: {', '.join(predicted) if predicted else '(none above threshold)'}")