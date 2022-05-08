# %%
import nltk
import numpy as np
from nltk.corpus import wordnet as wn

# %%
ss = wn.synset("cat.n.01")
for hyper in ss.hypernyms(): 
    print(ss)
    print(f'hyper: {str(hyper)}')
    print(hyper.name())
    ss = wn.synset(hyper.name())

# %%
s1 = wn.synset("cat.n.01")
s2 = wn.synset("tiger.n.02")
s3 = wn.synset("entity.n.01")

print(s1.min_depth())
print(s3.min_depth())

print(s1.shortest_path_distance(s2))
print(s1.lowest_common_hypernyms(s2))
print(s1.lowest_common_hypernyms(s2)[0].min_depth())


# %%
for ss in wn.synsets("tiger"):
    print(ss.name(), ss.lemma_names())
    print(f'def: {ss.definition()}')
    print(f'examples: {ss.examples()}')
    print("-----hyponyms-----")
    for hyp in ss.hyponyms():
        print(f'hyp: {str(hyp)}')
    print("-----hypernyms-----")
    for hyper in ss.hypernyms(): 
        print(f'hyper: {str(hyper)}')
    print("############\n")
# %%
matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix)
matrix = np.append(matrix,[[11,12, 13]], 0)
print(matrix)


# %%
print(wn.synset('dog.n.1').hypernym_paths())
print(max(len(hyp_path) for hyp_path in wn.synset('dog.n.1').hypernym_paths()))

# %%
max(max(len(path) for path in ss.hypernym_paths()) for ss in wn.all_synsets())

# %%
print(0/45)
# %%
matrix = np.empty((3,4),object)
matrix[0,1:] = ["a", "b", "c"]

print(matrix)
# %%
word1 = "cat"
word2 = "dog"

test_shortest_path_distance(word1,word2)

ancestor_s1 = set(s1[0].hypernyms())
ancestor_s2 = set(s2[0].hypernyms())