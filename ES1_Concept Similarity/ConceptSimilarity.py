import csv
import math

import numpy as np
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr, spearmanr
from tabulate import tabulate

# max(max(len(hyp_path) for hyp_path in ss.hypernym_paths()) for ss in wn.all_synsets())
MAX_DEPTH = 20
indices = np.empty((3, 4), object)
indices[0, 1:] = ["WUP", "PTH", "LCH"]
indices[1:, 0] = ["Pearson", "Spearman"]


# funzioni di IO

def load_data(filename):
    matrix = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        matrix.append(header)
        for row in csv_reader:
            matrix.append([row[0], row[1], float(row[2])])
        matrix = np.array(matrix, object)
    return matrix


def print_table(table, num_rows=None, show_indices=False):
    if num_rows:
        print(tabulate(table[:num_rows, :], headers='firstrow',
              showindex=show_indices, tablefmt='fancy_grid'))
    else:
        print(tabulate(table[:, :], headers='firstrow',
              showindex=show_indices, tablefmt='fancy_grid'))


# funzioni di navigazione degli alberi

def shortest_path_distance(word1, word2):
    """
    Gives the distance of the shortest path between the synsets of the two words.
    Returns:
        - None, if no ancestor nodes are common
        - 0, if the two words are identical
        - the distance of the shortest path, otherwise
    """
    if word1 == word2:
        return 0
    s1 = wn.synsets(word1)
    s2 = wn.synsets(word2)
    # Get the ancestor nodes of each synset
    ancestor_s1 = set()
    ancestor_s2 = set()
    print("###############################")
    for synset in s1:
        print(f'{synset.name()} - {synset.hypernym_paths()}')
        ancestor_s1.update(set(synset.hypernym_paths()))
    print(ancestor_s1)
    print("---------------------")
    for synset in s2:
        print(f'{synset.name()} - {synset.hypernyms()}')
        ancestor_s2.update(set(synset.hypernyms()))
    print(ancestor_s2)
    print("###############################")
    # Get the common ancestor nodes
    common_ancestors = ancestor_s1.intersection(ancestor_s2)
    # Get the shortest path distance between the common ancestor nodes
    if len(common_ancestors) == 0:
        return None
    else:
        return min([s1[0].shortest_path_distance(c) for c in common_ancestors])
    


# misure di similarità

def Wu_Palmer(term1, term2):
    """Wu & Palmer"""
    sim = 0
    for s1 in wn.synsets(term1):
        for s2 in wn.synsets(term2):
            lcs = s1.lowest_common_hypernyms(s2)
            for common_ancestor in lcs:
                nom = 2 * lcs[0].min_depth()
                den = s1.min_depth() + s2.min_depth()
                sim = max(sim, nom/den)
    return sim


def shortest_path(term1, term2):
    """Shortest path"""
    sim = 0
    for s1 in wn.synsets(term1):
        for s2 in wn.synsets(term2):
            len = s1.shortest_path_distance(s2)
            if len is None:
                len = 2 * MAX_DEPTH  # CHECK
            score = 2 * MAX_DEPTH - len
            sim = max(sim, score)
    return sim


def leakcock_chodorow(term1, term2):
    """Leakcock & Chodorow"""
    sim = 0
    for s1 in wn.synsets(term1):
        for s2 in wn.synsets(term2):
            den = 2 * MAX_DEPTH
            len = s1.shortest_path_distance(s2)
            if len is None:
                len = 2 * MAX_DEPTH
            elif len == 0:
                len = 1
                den += 1
            score = -1*math.log(len/den)
            sim = max(sim, score)
    return sim

# test algoritmo shortest_path_distance
def test_shortest_path_distance(word1, word2):
    print("mia funzione:", shortest_path_distance(word1, word2))
    for s1 in wn.synsets(word1):
        for s2 in wn.synsets(word2):
            print(f'{s1.name()} - {s2.name()}: {s1.shortest_path_distance(s2)}')



def get_scores():
    for idx, row in enumerate(data[1:, :]):
        word1, word2 = row[0], row[1]

        wup = Wu_Palmer(word1, word2)
        data[idx+1, 3] = wup

        pth = shortest_path(word1, word2)
        data[idx+1, 4] = pth

        lch = leakcock_chodorow(word1, word2)
        data[idx+1, 5] = lch
    #print(f'WUP_sim({s1},{s2}): {s1.wup_similarity(s2)}')


# Indici di correlazione
def get_indices():
    similarity_measures = range(1, 4)
    for col in similarity_measures:
        # Pearson
        corr, _ = pearsonr(data[1:, 2], data[1:, col+2])
        indices[1, col] = corr
        # Spearman
        corr, _ = spearmanr(data[1:, 2], data[1:, col+2])
        indices[2, col] = corr


data = load_data("WordSim353.csv")
scores = np.empty([354, 3], object)
scores[0] = ["WUP", "PTH", "LCH"]
data = np.append(data, scores, axis=1)


get_scores()
get_indices()
print_table(indices)
