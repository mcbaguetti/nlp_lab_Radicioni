import csv
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


def create_nasari():
    nasari = {}
    nasari_path = "data/mini_NASARI.tsv"

    with open(nasari_path, encoding="utf8") as file:
        tsv = csv.reader(file, delimiter="\t")

        for row in tsv:
            vector = []

            for i, elem in enumerate(row):
                if i == 0:
                    word = elem.split("__")[0]

                    if len(word.split("_(")) > 1:
                        word = word.split("_")[0]

                    if len(word.split("_")) > 1:
                        word = word.split("_")[0] + " " + word.split("_")[1]

                else:
                    vector.append(elem)

                nasari[word] = vector

    return nasari


def get_human_eval(file_path):
    means = []
    first_col = []
    second_col = []

    with open(file_path, encoding="utf8") as file:
        tsv = csv.reader(file, delimiter="\t")
        for line in tsv:

            first_col.append(float(line[2]))
            second_col.append(float(line[3]))
            avg = (float(line[2]) + float(line[3])) / 2
            means.append(avg)

        first = np.array(first_col, dtype=np.float32)
        second = np.array(second_col, dtype=np.float32)
        return means, pearsonr(first, second), \
               spearmanr(first, second), \
               cohen_kappa_score(first.astype(int), second.astype(int))


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


def get_babel_words(file_path):
    words_ita = []
    words_babel = {}

    with open(file_path, encoding="utf8") as file:
        tsv = csv.reader(file, delimiter="\t")
        for line in tsv:
            pair = [line[0], line[1]]
            words_ita.append(pair)
            words_babel[line[0]] = create_babel(line[0])
            words_babel[line[1]] = create_babel(line[1])

    return words_ita, words_babel


def cosine_sim(v1, v2):
    n_v1 = np.array(v1, dtype=float)
    n_v2 = np.array(v2, dtype=float)

    return np.dot(n_v1, n_v2) / (norm(n_v1) * norm(n_v2))


def get_similarity(words, babel_words, nasari):
    similarity = []

    for group in words:
        sim_score = 0
        if group[0] in babel_words and group[1] in babel_words:
            b1 = babel_words[group[0]]
            b2 = babel_words[group[1]]
            for syn1 in b1:
                for syn2 in b2:
                    if syn1 in nasari and syn2 in nasari:
                        sim_score = max(cosine_sim(nasari[syn1], nasari[syn2]), sim_score)

        similarity.append(sim_score)

    return similarity


def start():
    file_path = "../ES5_Semantic Similarity/data/it.test.data.tsv"

    nasari_dict = create_nasari()

    human_eval, pears, spear, cohen = get_human_eval(file_path)
    words_ita, babel_words = get_babel_words(file_path)
    similarity = get_similarity(words_ita, babel_words, nasari_dict)

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
    print("K Cohen Agreement: " + str(cohen))


start()
