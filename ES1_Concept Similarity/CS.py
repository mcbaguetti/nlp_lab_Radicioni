# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
import csv
from tabulate import tabulate

# %%
# max(max(len(hyp_path) for hyp_path in ss.hypernym_paths()) for ss in wn.all_synsets())
MAX_DEPTH = 20

# %%
def load_data(filename):
    matrix = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            matrix.append(row)
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            line_count += 1
        print(f'Processed {line_count} lines.')
        matrix = np.array(matrix)
    return matrix

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
                # if nom/den > sim:
                #     sim = nom/den
                #     print(sim, s1, s2)
    return sim

# Shortest path
def shortest_path(term1, term2):
    sim = 0
    for s1 in wn.synsets(term1):
        for s2 in wn.synsets(term2):
            len = s1.shortest_path_distance(s2)
            if len is None:
                len = 0
            score = 2 * MAX_DEPTH - len
            sim = max(sim, score)
    return sim

# Leakcock & Chodorowee

# %%
# Indici di correlazione


# %%
data = load_data('WordSim353.csv')
#print(tabulate(data[:27,:], headers='firstrow', showindex='always', tablefmt='fancy_grid'))
scores = np.empty([354,3], object)
scores[0] = ["WUP", "PTH", "LCH"]
data = np.append(data, scores, axis=1)
print(tabulate(data[:27,:], headers='firstrow', showindex='always', tablefmt='fancy_grid'))


# %%


# %%
s1 = wn.synset("cat.n.01")
s2 = wn.synset("tiger.n.02")
for idx, row in enumerate(data[1:,:]):
    word1 , word2 = row[0], row[1]

    wup = Wu_Palmer(word1, word2)
    data[idx+1, 3] = wup

    lch = shortest_path(word1, word2)
    data[idx+1, 4] = lch
#print(score)
#print(f'WUP_sim({s1},{s2}): {s1.wup_similarity(s2)}')
print(tabulate(data[:27,:], headers='firstrow', showindex='always', tablefmt='fancy_grid'))

# %%
