# %%
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
# %%
for ss in wn.synsets("proportionate"):
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
def sc2ss(sensekey):
    """
    Look up a synset given the information from SemCor
    """
    if wnsn != 0:
        return wn.lemma_from_key(sensekey).synset()
    else:
        pass # TODO
 
s1 = sc2ss('live%2:42:06::')
s2 = sc2ss('overall%5:00:00:gross:00')
ss = s2
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
# %% [markdown]
http://moin.delph-in.net/wiki/SemCor
https://www.nltk.org/api/nltk.corpus.reader.html#nltk.corpus.reader.semcor.SemcorCorpusReader
https://www.nltk.org/howto/corpus.html
