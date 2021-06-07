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
def print_info(ss):
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
    # TODO
    # if wnsn != 0:
    #     return wn.lemma_from_key(sensekey).synset()
    # else:
    #     pass 
    return wn.lemma_from_key(sensekey).synset()


# %%
s1 = sc2ss('live%2:42:06::')
s2 = sc2ss('overall%5:00:00:gross:00')
s3 = sc2ss('primary_election%1:04:00::')
ss = s3
# %%
sentences = semcor.xml('brown1/tagfiles/br-a01.xml').findall('context/p/s')
sent = sentences[0]
for word in sent:
    #print(word.tag, word.attrib["cmd"])
    if word.tag == "wf" and word.attrib["cmd"] == "done":
        if ("ot" not in word.attrib.keys()):
            lemma = word.attrib["lemma"]
            print('\033[1m' + lemma + '\033[0m')
            lexsn = word.attrib["lexsn"]
            ss = sc2ss(lemma + '%' + lexsn)
            print_info(ss)
    print()

# %%
a = []
a.append(1)
a.append(2)
a
# %% [markdown]
http://moin.delph-in.net/wiki/SemCor
https://www.nltk.org/api/nltk.corpus.reader.html#nltk.corpus.reader.semcor.SemcorCorpusReader
https://www.nltk.org/howto/corpus.html
