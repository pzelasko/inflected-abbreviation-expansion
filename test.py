import argparse
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from data import retrieve_words_to_abbreviate
from features import featurize, featurize_label, load_feature_converters

parser = argparse.ArgumentParser(
    description="Script for the inflected abbreviation expansion BLSTM model evaluation."
)
parser.add_argument("pickled_sentences")
parser.add_argument("features")
parser.add_argument("abbreviations")
parser.add_argument("input_model")
args = parser.parse_args()

words_with_abbreviations = retrieve_words_to_abbreviate(args.abbreviations)

with open(args.pickled_sentences, 'rb') as f:
    ncp_sentences = pickle.load(f)

features2idx, idx2feature, label2idx, idx2label = load_feature_converters(args.features)

eval_model = load_model(args.input_model)

y_eval = []
X_eval = []
errors = []
filt_ncp_sentences = []
for sent in ncp_sentences:
    try:
        X_eval.append(featurize(sent, features2idx))  # features => morphological tags of words in the sentence
        y_eval.append(featurize_label(sent[1][1], label2idx))  # target => morphological tags of abbreviable word
        filt_ncp_sentences.append(sent)
    except KeyError:
        errors.append(sent)

y_eval, X_eval = np.array(y_eval), pad_sequences(np.array(X_eval))

print(eval_model.metrics_names)
print(eval_model.evaluate(X_eval, y_eval, batch_size=1024))
