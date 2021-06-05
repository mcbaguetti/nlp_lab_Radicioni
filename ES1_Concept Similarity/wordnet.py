# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import nltk
from nltk.corpus import wordnet as wn


# %%
term = "board"

for ss in wn.synsets(term):
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
print(wn.synset("board.n.02").part_meronyms())
print(wn.synset("cat.n.01").part_meronyms())


# %%
s1 = wn.synset("cat.n.01")
s2 = wn.synset("dog.n.01")

print(f'def({s1}): {s1.definition()}\n')
print(f'def({s2}): {s2.definition()}\n')

print(f'WUP_sim({s1},{s2}): {s1.wup_similarity(s2)}')
print(f'LCH_sim({s1},{s2}): {s1.lch_similarity(s2)}')
print(f'PTH_sim({s1},{s2}): {s1.path_similarity(s2)}')
# %%
