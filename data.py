import gzip
from copy import copy
from itertools import chain
from pathlib import Path
from xml.etree.ElementTree import iterparse


def overlapping_grouper(iterable, n, prefix_symbol=('', ''), suffix_symbol=('', '')):
    queue = [prefix_symbol] * n
    offset = int(n // 2)
    for idx, item in enumerate(chain(iterable, [suffix_symbol] * offset)):
        queue.pop(0)
        queue.append(item)
        if idx >= offset:
            yield copy(queue)


def retrieve_words_to_abbreviate(path):
    with open(path) as expansions_file:
        expansions = {parts[0]: parts[1:] for parts in map(lambda l: l.strip().split(';'), expansions_file)}
    return frozenset(chain.from_iterable(expansions.values()))


def find_morphosyntax_files(path):
    path = Path(path)
    return [path] if path.is_file() else path.glob('*/**/ann_morphosyntax.xml*')


def parse_annotated_sentences(annotation_xml):

    # Helper functions to retrieve the XML sub-element with the same name for a given element
    # Doesn't seem readable at first but it's the same as the XML

    def orth(fs_elem):
        return fs_elem[0][0].text

    def pos(fs_elem):
        disamb = [e for e in fs_elem if e.get('name', '') == 'disamb'][0]
        return disamb[0][1][0].text.split(':', maxsplit=1)[1]

    def fs(seg_elem):
        return seg_elem[0]

    opener = gzip.open if annotation_xml.suffix == '.gz' else open
    with opener(annotation_xml, encoding='utf-8') as ann_f:
        for event, elem in iterparse(ann_f):
            if tag_uri_and_name(elem)[1] == 's':  # just parsed a sentence
                sent = [(orth(fs(seg)), pos(fs(seg))) for seg in list(elem)]
                yield sent


def sentences_with_abbreviations(corpus_path, abbreviable_words):
    for f in find_morphosyntax_files(corpus_path):
        for sentence in parse_annotated_sentences(f):
            target_abbreviations = ((word, tag, pos) for pos, (word, tag) in enumerate(sentence) if
                                  word.lower() in abbreviable_words)
            for abbreviation in target_abbreviations:
                yield sentence, abbreviation


def tag_uri_and_name(elem):
    """https://stackoverflow.com/questions/1953761/accessing-xmlns-attribute-with-python-elementree"""
    if elem.tag[0] == "{":
        uri, ignore, tag = elem.tag[1:].partition("}")
    else:
        uri = None
        tag = elem.tag
    return uri, tag
