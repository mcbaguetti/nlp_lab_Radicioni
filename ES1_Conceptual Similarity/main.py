from collections import deque
import math
import utils

import numpy as np
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr, spearmanr


MAX_DEPTH = 20 # max(max(len(hyp_path) for hyp_path in ss.hypernym_paths()) for ss in wn.all_synsets())
indices = np.empty((3, 4), object)
indices[0, 1:] = ["WUP", "PTH", "LCH"]
indices[1:, 0] = ["Pearson", "Spearman"]


# funzioni di navigazione degli alberi

def hypernym_depth(sense):
    """
    Gives a dictionary of the hypernyms of the given synset,
    with the corresponding depth.
    """
    sense_depth_queue = deque([(sense, 0)]) # for efficiency
    sense_depth_dict = {}
    while sense_depth_queue:
        s, depth = sense_depth_queue.popleft()
        if s in sense_depth_dict:
            continue
        sense_depth_dict[s] = depth
        depth += 1
        for hypernym in s.hypernyms():
            sense_depth_queue.extend([(hypernym, depth)])

    return sense_depth_dict

def shortest_path_distance(sense1, sense2):
    """
    Gives the distance of the shortest path between the synsets of the two words.
    Returns:
        - 0, if the two words are identical
        - the distance of the shortest path, otherwise
    """
    
    if sense1.name() == sense2.name():
        return 0

    hyp_dict1 = hypernym_depth(sense1)
    hyp_dict2 = hypernym_depth(sense2)
    min_dist = 2 * MAX_DEPTH

    for sense, dist1 in hyp_dict1.items():
        dist2 = hyp_dict2.get(sense, 2 * MAX_DEPTH)
        min_dist = min(min_dist, dist1 + dist2)
    
    return min_dist
    
def lowest_common_subsumer(common_hypernyms):
    """
    Returns the lowest common subsumer of the given synsets.
    """
    lowest_common_distance = 0
    lowest_common_hypernym = None
    for hypernym in common_hypernyms:
        # calculate the maximum depth of a synset
        hypernym_path = hypernym.hypernym_paths()
        max_depth = max(len(path) for path in hypernym_path)

        if max_depth > lowest_common_distance:
            lowest_common_distance = max_depth
            lowest_common_hypernym = hypernym
    return lowest_common_hypernym


# misure di similarità

def Wu_Palmer(synset1, synset2):
    """Wu & Palmer"""
    max_sim = 0
    for s1 in synset1:
        hypernym_path1 = s1.hypernym_paths()
        for s2 in synset2:
            hypernym_path2 = s2.hypernym_paths()
            # set of common hypernyms between the paths
            hypernyms1 = [y for x in hypernym_path1 for y in x]
            hypernyms2 = [y for x in hypernym_path2 for y in x]
            common_hypernyms = set(hypernyms1).intersection(set(hypernyms2))
            # get the lowest common hypernym
            lcs = lowest_common_subsumer(common_hypernyms)

            if lcs is None:
                sim = 0
            else:
                lcd = lcs.min_depth()
                if lcd == 0:
                    lcd = 1
                nom = 2 * lcd
                den = s1.min_depth() + s2.min_depth()
                sim = nom / den
            max_sim = max(max_sim, sim)

    return max_sim


def shortest_path(synset1, synset2):
    """Shortest path"""
    sim = 0
    distances = []
    for s1 in synset1:
        for s2 in synset2:
            dist = shortest_path_distance(s1,s2)
            distances.append(dist)
    if len(distances) > 0:
        sim = min(distances)
        sim = ( 2 * MAX_DEPTH - sim ) / ( 2 * MAX_DEPTH )

    return sim


def leakcock_chodorow(synset1, synset2):
    """Leakcock & Chodorow"""
    sim = 0
    distances = []
    for s1 in synset1:
        for s2 in synset2:
            dist = shortest_path_distance(s1,s2)
            distances.append(dist)
    if len(distances) > 0:
        sim = min(distances)
        if sim == 0:
            sim = 1
        sim = - (math.log(sim / (2 * MAX_DEPTH))) / (math.log(2 * MAX_DEPTH + 1))
    return sim

# Punteggi e indici di correlazione
def get_scores():
    for idx, row in enumerate(data[1:, :]):
        word1, word2 = row[0], row[1]
        synset1 = wn.synsets(word1)
        synset2 = wn.synsets(word2)

        wup = Wu_Palmer(synset1, synset2)
        data[idx+1, 3] = wup

        pth = shortest_path(synset1, synset2)
        data[idx+1, 4] = pth

        lch = leakcock_chodorow(synset1, synset2)
        data[idx+1, 5] = lch

def get_indices():
    similarity_measures = range(1, 4)
    for col in similarity_measures:
        # Pearson
        corr, _ = pearsonr(data[1:, 2], data[1:, col+2])
        indices[1, col] = corr
        # Spearman
        corr, _ = spearmanr(data[1:, 2], data[1:, col+2])
        indices[2, col] = corr

data = utils.load_data("WordSim353.csv")
scores = np.empty([354, 3], object)
scores[0] = ["WUP", "PTH", "LCH"]
data = np.append(data, scores, axis=1)

get_scores()
get_indices()
utils.print_table(indices)
