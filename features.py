# Represent each sentence as a sequence of numbers corresponding to morphosyntactic tag of each word
import pickle

import numpy as np


def featurize(sent, features2idx):
    pos = sent[1][2]
    feats = np.zeros((len(sent[0])), dtype=np.int32)
    for idx, (word, tags) in enumerate(sent[0]):
        if idx == pos:
            feats[idx] = features2idx['UNK']
        else:
            feats[idx] = features2idx[tags]
    return feats


def featurize_label(tags, label2idx):
    ohe_label = np.zeros(len(label2idx), dtype=np.int32)
    label_idx = label2idx.get(tags, len(label2idx) - 1)
    ohe_label[label_idx] = 1
    return ohe_label


def make_feature_converters(prepared_sentences, dump_path=None):
    # Find all the tags that appear in the inputs (features) and the outputs (target_tags)
    features = list(sorted(set(item[1] for s in prepared_sentences for item in s[0]))) + ['UNK']
    target_tags = list(sorted(set(s[1][1] for s in prepared_sentences)))
    print('Unique tag combinations in corpus [input features]:', len(features))
    print('Unique abbreviation tag combinations in corpus [labels]:', len(target_tags))

    # Create mapping "tag -> id"
    features2idx = dict(zip(features, range(len(features))))
    idx2features = features

    # Reverse mapping "id -> tag"
    label2idx = dict(zip(target_tags, range(len(target_tags))))
    idx2label = target_tags

    if dump_path is not None:
        with open(dump_path, 'wb') as f:
            pickle.dump((features2idx, idx2features, label2idx, idx2label), f)

    return features2idx, idx2features, label2idx, idx2label


def load_feature_converters(path):
    with open(path, 'rb') as f:
        return pickle.load(f)