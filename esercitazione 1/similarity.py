from nltk.corpus import wordnet as wn
from math import log
import util


def word_to_synset(w1, w2):

    max_wu = 0.0
    max_sh = 0.0
    max_lc = 0.0

    for syn1 in wn.synsets(w1):
        for syn2 in wn.synsets(w2):
            wu_ans = wu(syn1, syn2)
            sh_ans = sh_path(syn1, syn2)
            lc_ans = leak(syn1, syn2)

            max_wu = max(max_wu, wu_ans)
            max_sh = max(max_sh, sh_ans)
            max_lc = max(max_lc, lc_ans)

    return max_wu, max_sh, max_lc


def wu(syn1, syn2):
    util.depth(syn1)
    # return 2 * util.depth(util.lcs(syn1, syn2)) / (util.depth(syn1) + util.depth(syn2))


def sh_path(syn1, syn2):
    # return 2 * util.depth_max() - util.distance(syn1, syn2)
    return


def leak(syn1, syn2):
    # return -log(util.distance(syn1, syn2) / util.depth_max())
    return
