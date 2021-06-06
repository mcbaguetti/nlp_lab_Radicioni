# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
import csv
import math
from tabulate import tabulate
from scipy.stats import pearsonr, spearmanr

# %%
MAX_DEPTH = 20 # max(max(len(hyp_path) for hyp_path in ss.hypernym_paths()) for ss in wn.all_synsets())
indices = np.empty((3,4), object)
indices[0,1:] = ["WUP", "PTH", "LCH"]
indices[1:,0] = ["Pearson", "Spearman"]

# %%
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

# %%
def print_table(table, num_rows=None, show_indices=False):
    if num_rows:
        print(tabulate(table[:num_rows,:], headers='firstrow', showindex=show_indices, tablefmt='fancy_grid'))
    else:
        print(tabulate(table[:,:], headers='firstrow', showindex=show_indices, tablefmt='fancy_grid'))

# %%
# Misure di similarità

# Wu & Palmer
def Wu_Palmer(term1, term2):
    sim = 0
    for s1 in wn.synsets(term1):
        for s2 in wn.synsets(term2):
            lcs = s1.lowest_common_hypernyms(s2)
            for common_ancestor in lcs:
                nom = 2 * lcs[0].min_depth()
                den = s1.min_depth() + s2.min_depth()
                sim = max(sim, nom/den)
    return sim

# Shortest path
def shortest_path(term1, term2):
    sim = 0
    for s1 in wn.synsets(term1):
        for s2 in wn.synsets(term2):
            len = s1.shortest_path_distance(s2)
            if len is None:
                len = 2 * MAX_DEPTH # CHECK
            score = 2 * MAX_DEPTH - len
            sim = max(sim, score)
    return sim

# Leakcock & Chodorow
def leakcock_chodorow(term1, term2):
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

# %%
def get_scores():
    for idx, row in enumerate(data[1:,:]):
        word1 , word2 = row[0], row[1]

        wup = Wu_Palmer(word1, word2)
        data[idx+1, 3] = wup

        pth = shortest_path(word1, word2)
        data[idx+1, 4] = pth

        lch = leakcock_chodorow(word1, word2)
        data[idx+1, 5] = lch
    #print(f'WUP_sim({s1},{s2}): {s1.wup_similarity(s2)}')

# %%
# Indici di correlazione
def get_indices():
    similarity_measures = range(1,4)
    for col in similarity_measures:
        # Pearson
        corr, _ = pearsonr(data[1:,2], data[1:,col+2])
        indices[1,col] = corr
        # Spearman
        corr, _ = spearmanr(data[1:,2], data[1:,col+2])
        indices[2,col] = corr

# %%
data = load_data('WordSim353.csv')
scores = np.empty([354,3], object)
scores[0] = ["WUP", "PTH", "LCH"]
data = np.append(data, scores, axis=1)

# %%
get_scores()
#print_table(data, 27, True)
get_indices()
print_table(indices)

# %%
