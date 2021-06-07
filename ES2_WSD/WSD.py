# %%
import nltk
from nltk.corpus import semcor
import random

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
sentences = semcor.xml('brown1/tagfiles/br-a01.xml').findall('context/p/s')
sent = sentences[0]

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
def get_words(sentence):
    words = []
    for word in sentence:
        #print(word.tag, word.attrib["cmd"])
        if word.tag == "wf" and word.attrib["cmd"] == "done":
            if ("ot" not in word.attrib.keys()):
                lemma = word.attrib["lemma"]
                words.append(lemma)
                print('\033[1m' + lemma + '\033[0m')
                lexsn = word.attrib["lexsn"]
                ss = sc2ss(lemma + '%' + lexsn)
                print_info(ss)
                print()
    return words

# %%
def Lesk(word, sentence):
    best_sense = "a" # TODO
    max_overlap = 0
    context = get_words(sentence)
    for ss in wn.synsets(word):
        signature = [] # TODO
        overlap = compute_overlap(signature, context)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = ss
    return best_sense

# %%
semcor.words()

# %%
semcor.chunks()
# %%
semcor.sents()
# %%
c_sents = semcor.chunk_sents()
c_sents[9]

# %%
random_list = []
for i in range(0, 50):
  # random numbers from 0 to len(semcor.chunk_sents())
    random_list.append(random.randint(0, 37176)) 
random_list
# %%
