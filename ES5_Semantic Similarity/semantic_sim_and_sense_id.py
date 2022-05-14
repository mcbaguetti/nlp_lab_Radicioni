import csv
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


# create nasari dictionary with babelnet id and their vectors
def create_nasari():
    nasari = {}
    word_sense = {}  # dictionary with babelnet id and associated sense
    nasari_path = "data/mini_NASARI.tsv"

    with open(nasari_path, encoding="utf8") as file:
        tsv = csv.reader(file, delimiter="\t")

        for row in tsv:
            vector = []

            for i, elem in enumerate(row):
                if i == 0:
                    word = elem.split("__")[0]
                    sense = elem.split("__")[1]
                    if len(word.split("_(")) > 1:
                        word = word.split("_")[0]

                    if len(word.split("_")) > 1:
                        word = word.split("_")[0] + " " + word.split("_")[1]

                    word_sense[word] = sense

                else:
                    vector.append(elem)

                nasari[word] = vector

    return nasari, word_sense


# find the avg human evaluation of the pairs, write them down and calc inter rate agreement
def get_human_eval(file_path):

    new_file_path = "out/avg_eval.tsv"

    means = []
    first_col = []
    second_col = []

    with open(file_path, encoding="utf8") as file:
        with open(new_file_path, encoding="utf-8", mode='w') as new_file:
            writ = csv.writer(new_file, delimiter="\t")
            tsv = csv.reader(file, delimiter="\t")
            for line in tsv:
                first_col.append(float(line[2]))
                second_col.append(float(line[3]))
                avg = (float(line[2]) + float(line[3])) / 2
                means.append(avg)
                line.append(str(avg))
                writ.writerow(line)

            first = np.array(first_col, dtype=np.float32)
            second = np.array(second_col, dtype=np.float32)

    return means, pearsonr(first, second), \
            spearmanr(first, second), \
            cohen_kappa_score(first.astype(int), second.astype(int))


# given a word it returns its babelnet ids
def create_babel(word):
    words_syn_path = "../ES5_Semantic Similarity/data/SemEval17_IT_senses2synsets.txt"
    babel_syns_found = False
    vectors = []

    with open(words_syn_path, encoding="utf8") as file:
        txt = csv.reader(file)
        for line in txt:
            for l in line:
                if l[0] == "#":
                    if babel_syns_found:
                        babel_syns_found = False

                    if word == l.split("#")[1]:
                        babel_syns_found = True

                elif babel_syns_found:
                    vectors.append(l)
    return vectors


# for each word in the annotated file it creates a dict and a list
def get_babel_words(file_path):
    words_ita = []  # list with italian word pairs
    words_babel = {}  # dict with italian word : [babelnet-id-1, babelnet-id-2, etc]

    with open(file_path, encoding="utf8") as file:
        tsv = csv.reader(file, delimiter="\t")
        for line in tsv:
            pair = [line[0], line[1]]
            words_ita.append(pair)
            words_babel[line[0]] = create_babel(line[0])
            words_babel[line[1]] = create_babel(line[1])

    return words_ita, words_babel


# calculates cosine similarity
def cosine_sim(v1, v2):
    n_v1 = np.array(v1, dtype=float)
    n_v2 = np.array(v2, dtype=float)

    return np.dot(n_v1, n_v2) / (norm(n_v1) * norm(n_v2))


# calculates the max similarity and find the senses that max the similarity
def get_similarity(words, babel_words, nasari):
    similarity = []
    synsets = [[] for _ in range(51)]  # at max we have 50 pairs of synsets

    for i, group in enumerate(words):
        sim_score = 0
        if group[0] in babel_words and group[1] in babel_words:
            b1 = babel_words[group[0]]
            b2 = babel_words[group[1]]
            for syn1 in b1:
                for syn2 in b2:
                    if syn1 in nasari and syn2 in nasari:
                        cos = cosine_sim(nasari[syn1], nasari[syn2])
                        if sim_score < cos:
                            sim_score = cos
                            synsets[i] = (syn1, syn2)

        similarity.append(sim_score)

    return similarity, synsets


# for each babelnet id found, the function associates the correlated terms (taken from babelnet by hand)
def identify_terms(synsets, word_sense):
    file_path = "./data/babel_sense.txt"
    new_file_path = "out/sense_identification.tsv"

    with open(file_path, encoding="utf8") as file:
        with open(new_file_path, encoding="utf-8", mode='w') as new_file:
            for syn, w, row in zip(synsets, word_sense, file):
                if syn and w:
                    s = syn[0] + "\t" + syn[1] + "\t" + w[0] + "\t" + w[1] + "\t" + row
                    new_file.write(s)

    return


# get babelnet_ids annotated by hand
def get_babelnet_annotated(path):
    b_ids = []

    with open(path, encoding="utf8") as file:
        for row in file:
            b_id2 = row.split("\t")[1]
            b_ids.append((row.split("\t")[0], b_id2.split("\n")[0]))

    return b_ids


# find accuracy between pair and single human chosen bnet-ids and computer chosen bnet-ids
def find_accuracy(computer_syns, human_syns):

    single = 0
    pairs = 0
    tot_single = len(human_syns) * 2
    tot_pair = len(human_syns)

    for c, h in zip(computer_syns, human_syns):
        if c == h:
            pairs += 1

        for id1, id2 in zip(c, h):
            if id1 == id2:
                single += 1

    return single/tot_single, pairs/tot_pair


# function that handles semantic similarity and sense identification
def start():
    file_path = "../ES5_Semantic Similarity/data/it.test.data.tsv"
    annot_syns_path = "./data/annotated_syns.txt"

    nasari_dict, word_sense = create_nasari()
    human_eval, pears, spear, cohen = get_human_eval(file_path)
    words_ita, babel_words = get_babel_words(file_path)
    similarity, synsets = get_similarity(words_ita, babel_words, nasari_dict)

    print("Consegna 1: Semantic Similarity")
    print("Avg human evaluation: ")
    print(human_eval)

    print("\n")
    print("Pearson Inter Rate Agreement: " + str(pears))
    print("Spearman Inter Rate Agreement: " + str(spear))

    print("\n")
    print("Cosine and Nasari based similarity: ")
    print(similarity)

    print("\n")
    print("Pearson Evaluation: ")
    print(pearsonr(human_eval, similarity))
    print("Spearman Evaluation: ")
    print(spearmanr(human_eval, similarity))

    print("\n")
    print("Consegna 2: Sense Identification")
    print("K Cohen Agreement: " + str(cohen))
    annotated_syns = get_babelnet_annotated(annot_syns_path)
    identify_terms(annotated_syns, words_ita)
    single_acc, pair_acc = find_accuracy(synsets, annotated_syns)
    print("Single Accuracy: " + str(single_acc))
    print("Pair Accuracy: " + str(pair_acc))


start()
