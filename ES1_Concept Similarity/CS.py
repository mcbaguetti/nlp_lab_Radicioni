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
    print("----------")


# %%



