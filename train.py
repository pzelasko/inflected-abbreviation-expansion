import random
import pickle
import gzip
import argparse

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from features import featurize_label, featurize, make_feature_converters

parser = argparse.ArgumentParser(
    description="Script for training the inflected abbreviation expansion BLSTM model."
)
parser.add_argument("pickled_sentences")
parser.add_argument("output_features")
parser.add_argument("output_model")
args = parser.parse_args()

# Load the train/valid dataset
opener = gzip.open if args.pickled_sentences.endswith('.gz') else open
with opener(args.pickled_sentences, 'rb') as sents_f:
    interesting_sents = pickle.load(sents_f)
random.shuffle(interesting_sents)

features2idx, idx2feature, label2idx, idx2label = make_feature_converters(
    interesting_sents,
    dump_path=args.output_features
)

# Convert the textual dataset to one-hot-encoded features
y = []
X = []
for sent in interesting_sents:
    # Skip longer sentences to speed up the computation (could use bucketing)
    if len(sent[0]) > 30:
        continue
    X.append(featurize(sent, features2idx))  # features => morphological tags of words in the sentence
    y.append(featurize_label(sent[1][1], label2idx))  # target => morphological tags of abbreviable word
y, X = np.array(y), pad_sequences(np.array(X))

# Construct the BLSTM model
L2 = 0.0005

input_layer = Input(shape=(None,))
x = Embedding(len(features2idx), 64)(input_layer)
x = Bidirectional(LSTM(128, dropout=0.2, return_sequences=True))(x)
x = Bidirectional(LSTM(128, dropout=0.2, return_sequences=True))(x)
x = Bidirectional(LSTM(128, dropout=0.2))(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(L2))(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(idx2label), activation='softmax', kernel_regularizer=l2(L2))(x)

model = Model(input_layer, output_layer)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    X, y,
    batch_size=2048,
    validation_split=0.01,
    epochs=200,
    verbose=1,
    callbacks=[
        TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True, write_images=False),
        ModelCheckpoint(args.output_model, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0, mode='auto', epsilon=0.0001,
                          cooldown=2, min_lr=0),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto')
    ]
)
