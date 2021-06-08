# %%
import nltk
import numpy as np
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
import os
import random
# %%
for ss in wn.synsets("bank"):
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
directories = ["brown1", "brown2", "brownv"]
random_index = random.randrange(50000) % len(directories)
local_path = directories[random_index] + '/tagfiles'
abs_path = '/home/xps/nltk_data/corpora/semcor/' + local_path
print(random_index, local_path, abs_path)
#print(len(next(os.walk(path))[2]))
#print((next(os.walk(abs_path))[2]))
random_index = random.randrange(50000) % len(next(os.walk(abs_path))[2])
current_file = next(os.walk(abs_path))[2][random_index]
print(random_index, current_file)
print(local_path+'/'+current_file)
# path = '/home/xps/nltk_data/corpora/semcor/brown1'
# for root, directories, files in os.walk(path):
# 	for name in files:
# 		print("file",os.path.join(root, name))
# 	for name in directories:
# 		print("dir",os.path.join(root, name))

# %%
sentences = semcor.xml('brown1/tagfiles/br-a01.xml').findall('context/p/s')
#sentences = semcor.xml(local_path+'/'+current_file).findall('context/p/s')
sent = sentences[0]
for word in sent:
    print(word.text)
    if word.tag == "wf" and word.attrib["cmd"] == "done":
        if ("ot" not in word.attrib.keys()):
            lemma = word.attrib["lemma"]
            print('\033[1m' + lemma + '\033[0m')
            # lexsn = word.attrib["lexsn"]
            # ss = sc2ss(lemma + '%' + lexsn)
            # print_info(ss)
    print()

# %%
ss=wn.synset("bank.n.01")
a = set()
a.update([1,2])
a.update([3,4,8])
b = set([4,8])
c = a.intersection(b)
len(c)
# %% [markdown]
http://moin.delph-in.net/wiki/SemCor
github.com/Akirato/Lesk-Algorithm/blob/master/leskAlgorithm.py
https://www.nltk.org/api/nltk.corpus.reader.html#nltk.corpus.reader.semcor.SemcorCorpusReader
https://www.nltk.org/howto/corpus.html

# %%
semcor.words()

# %%
semcor.chunks()
# %%
semcor.sents()
# %%
c_sents = semcor.chunk_sents()
c_sents
# %%
try:
  ss = wn.lemma_from_key('tell%2:32:01::').synset()
  print(ss)
except:
    # No synset found for key
  print("An exception occurred")
# %%
a = set([1,2,3,4,5])
a.difference([4,5])
# %%
word = 'manometer'
for ss in wn.synsets(word):
    print_info(ss)
# %%
