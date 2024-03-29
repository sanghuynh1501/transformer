from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {"SOS": 0, "EOS": 1, "PAD": 2}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = s.replace(" .", "")
    s = s.replace(".", "")
    return s


def filter_pair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length


def filter_pairs(pairs, max_length):
    return [pair for pair in pairs if filter_pair(pair, max_length)]


def read_languages(lang1, lang2, reverse=False):
    print("Reading lines...")

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepare_data(lang1, lang2, max_length=None, reverse=False):
    input_lang, output_lang, pairs = read_languages(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    if max_length is not None:
        pairs = filter_pairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("len(pairs) ", len(pairs))
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
