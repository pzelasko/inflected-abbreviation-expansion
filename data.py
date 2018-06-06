import os
import gzip
from itertools import chain


def overlapping_grouper(iterable, n, prefix_symbol=('', ''), suffix_symbol=('', '')):
    from itertools import chain
    from copy import copy
    queue = [prefix_symbol] * n
    offset = int(n // 2)
    for idx, item in enumerate(chain(iterable, [suffix_symbol] * offset)):
        queue.pop(0)
        queue.append(item)
        if idx >= offset:
            yield copy(queue)


def retrieve_abbreviable_words(path):
    with open(path, encoding='utf8') as exp_f:
        expansions = {parts[0]: parts[1:] for parts in map(lambda l: l.strip().split(';'), exp_f)}
    return set(chain.from_iterable(expansions.values()))


def scan_morphosyntax_xmls(path):
    return [path] if os.path.isfile(path) else (
        os.path.join(root, f) for root, _, fs in os.walk(path)
        for f in fs
        if 'ann_morphosyntax.xml' in f
    )


def process_file_streaming_sents(annotation_xml):
    from xml.etree.ElementTree import iterparse

    def orth(fs_elem):
        # fs_elem => <fs type="morph">
        # return: fs => <f name="orth"> => <string> => text
        return fs_elem[0][0].text

    def pos(fs_elem):
        # fs_elem => <fs type="morph">
        # return: fs => <f name="disamb"> => <fs type="tool_report"> => <f name="interpretation> => <string> => text
        disamb = [e for e in fs_elem if e.get('name', '') == 'disamb'][0]
        return disamb[0][1][0].text.split(':', maxsplit=1)[1]

    def fs(seg_elem):
        return seg_elem[0]

    opener = gzip.open if annotation_xml.endswith('.gz') else open
    with opener(annotation_xml, encoding='utf-8') as ann_f:
        for event, elem in iterparse(ann_f):
            if tag_uri_and_name(elem)[1] == 's':  # just parsed a sentence
                sent = [(orth(fs(seg)), pos(fs(seg))) for seg in list(elem)]
                yield sent


def make_sents_iterator(corpus_path, abbreviable_words):
    for f in scan_morphosyntax_xmls(corpus_path):
        for sent in process_file_streaming_sents(f):
            interesting_tuples = ((word, tag, pos) for pos, (word, tag) in enumerate(sent) if
                                  word.lower() in abbreviable_words)
            for t in interesting_tuples:
                yield sent, t


def tag_uri_and_name(elem):
    """https://stackoverflow.com/questions/1953761/accessing-xmlns-attribute-with-python-elementree"""
    if elem.tag[0] == "{":
        uri, ignore, tag = elem.tag[1:].partition("}")
    else:
        uri = None
        tag = elem.tag
    return uri, tag
