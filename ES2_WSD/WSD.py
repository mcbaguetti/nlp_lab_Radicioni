# %%
import nltk
from nltk.corpus import semcor
import random
import re

# %%
def sc2ss(sensekey):
    """
    Look up a synset given the information from SemCor
    """
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

def get_random_set():
    random_set = set()
    while len(random_set) != 50:
    # random numbers from 0 to len(semcor.chunk_sents())
        random_set.add(random.randint(0, 37176)) 
    return random_set

# %%
def get_context(sentence):
    words = set()
    # for word in sentence:
    #     #print(word.tag, word.attrib["cmd"])
    #     if word.tag == "wf" and word.attrib["cmd"] == "done":
    #         if ("ot" not in word.attrib.keys()):
    #             lemma = word.attrib["lemma"]
    #             words.add(lemma)
    #             print('\033[1m' + lemma + '\033[0m')
    #             lexsn = word.attrib["lexsn"]
    #             wnsn = word.attrib["wnsn"]
    #             if wnsn != "0":
    #                 ss = sc2ss(lemma + '%' + lexsn)
    #                 print_info(ss)
    #             print()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    words = set(sentence.split())
    return words


# %%
def get_signature(sense):
    gloss = sense.definition()
    gloss = re.sub(r'[^\w\s]', '', gloss)
    gloss_set = set(gloss.split())

    examples = sense.examples()
    examples_set = set()
    for ex in examples:
        ex = re.sub(r'[^\w\s]', '', ex)
        examples_set.update(ex.split())
    
    return gloss_set.union(examples_set)

# %%
def compute_overlap(signature, context):
    return len(signature.intersection(context))

# %%
def Lesk(word, sentence):
    best_sense = None # CHECK
    max_overlap = 0
    context = get_context(sentence)
    for ss in wn.synsets(word):
        signature = get_signature(ss)
        overlap = compute_overlap(signature, context)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = ss
    return best_sense


# %%
word = "bank"
sentence = "the bank can guarantee deposits will eventually cover future tuition costs because it invests in adjustable-rate mortgage securities."
ss = Lesk(word, sentence)
ss
# %%
