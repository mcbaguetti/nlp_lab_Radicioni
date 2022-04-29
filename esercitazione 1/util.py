from nltk.corpus.reader.wordnet import Synset


def depth(syn: Synset):
    dep = 0
    hyper = syn.hypernyms()
    print(syn)
    print(hyper)
    if hyper:
        for h in hyper:
            print(h)

    return dep


def lcs(syn1, syn2):
    return 0


def depth_max():
    return 0


def distance(syn1, syn2):
    return 0

