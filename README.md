# Inflected Abbreviation Expansion

Complementary repository for the paper ["Expanding Abbreviations in a Strongly Inflected Language: Are Morphosyntactic Tags Sufficient?"](https://arxiv.org/abs/1708.05992) presented at LREC 2018

## Requirements

- numpy 1.14.1
- keras 2.1.6
- tensorflow 1.8.0

## Usage

- Download [PSC](http://clip.ipipan.waw.pl/PSC) and [NCP](http://clip.ipipan.waw.pl/NationalCorpusOfPolish) datasets.

- run `prepare_dataset.py` on both datasets (might take a while)

- the default abbreviation set with acceptable expansions in provided in file `abbreviations_list.txt`

- run `train.py` 

- run `test.py`

## Caveats

Have in mind that this recipe will not use the WCRFT morphosyntactic tags as mentioned in the paper, as it requires a significant effort to prepare a portable recipe with this feature. Instead, this recipe uses default morphosyntactic tags provided with the PSC corpus. If you'd like to use WCRFT anyway, [WCRFT can be found here](http://nlp.pwr.wroc.pl/redmine/projects/wcrft/wiki).