# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
import csv
from tabulate import tabulate

# %%
with open('WordSim353.csv') as csv_file:
    matrix = []
    csv_reader = csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:
        matrix.append(row)
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
        line_count += 1
    print(f'Processed {line_count} lines.')
    matrix = np.array(matrix)
    print(tabulate(matrix[:27,:], headers='firstrow', showindex='always', tablefmt='fancy_grid'))
    print(matrix[1])

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


# Leakcock & Chodorowee

# %%
# Indici di correlazione


# %%
s1 = wn.synset("cat.n.01")
s2 = wn.synset("tiger.n.02")
score = Wu_Palmer("cat","tiger")
print(score)
print(f'WUP_sim({s1},{s2}): {s1.wup_similarity(s2)}')

# %%
