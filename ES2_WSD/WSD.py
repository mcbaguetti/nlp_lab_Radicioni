# %%
import nltk
from nltk.corpus import semcor
import random

# %%
sent = semcor.xml('brown1/tagfiles/br-a01.xml').findall('context/p/s')[0]
for wordform in sent.getchildren():
    print(wordform.text, end=' ')
    for key in sorted(wordform.keys()):
        print(key + '=' + wordform.get(key), end=' ')
    print()

# %%
semcor.words()

# %%
semcor.chunks()
# %%
semcor.sents()
# %%
c_sents = semcor.chunk_sents()
c_sents[50]

# %%
random_list = []
for i in range(0, 50):
  # random numbers from 0 to len(semcor.chunk_sents())
    random_list.append(random.randint(0, 37176)) 
random_list
# %%
