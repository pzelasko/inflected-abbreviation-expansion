import argparse
import pickle

from data import make_sents_iterator, retrieve_abbreviable_words

parser = argparse.ArgumentParser(
    description="Read decompressed NCP (NKJP) or PSC dataset directory "
                "and extract sentences to use in abbreviation expansion modeling."
)
parser.add_argument("dataset_path")
parser.add_argument("abbreviations_list")
parser.add_argument("output")
args = parser.parse_args()

sentences = list(
    make_sents_iterator(
        args.dataset_path,
        retrieve_abbreviable_words(args.abbreviations_list)
    )
)

with open(args.output, 'wb') as f:
    pickle.dump(sentences, f)
