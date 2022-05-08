# %%
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
import re
import os
import random

random.seed(1997) # TODO: togliere
stop_words = []
with open('stop_words_FULL.txt', 'r') as f:
    for line in f:
        stop_words.append(line.strip())
print(stop_words)
# %%
def sc2ss(sensekey):
    """
    Look up a synset given the information from SemCor
    """
    try:
        return wn.lemma_from_key(sensekey).synset()
    except: # No synset found for key (es: 'previous%5:00:00:preceding(a):00')
        return None

def get_random_sentence():
    rand_int = random.randrange(50000)
    directories = ["brown1", "brown2", "brownv"]

    random_index = rand_int % len(directories)
    local_path = directories[random_index] + '/tagfiles'
    abs_path = '/home/apo/nltk_data/corpora/semcor/' + local_path

    random_index = rand_int % len(next(os.walk(abs_path))[2])
    current_file = next(os.walk(abs_path))[2][random_index]
    sentences = semcor.xml(local_path+'/'+current_file).findall('context/p/s')
    sc_sent = sentences[rand_int % len(sentences)]

    return sc_sent

def get_random_word(sentence):
    found = False
    iterations = 0
    result = {}
    while iterations < len(sentence) and not found:
        rand_int = random.randrange(50000)
        word = sentence[rand_int % len(sentence)]
        if word.tag == "wf" and word.attrib["cmd"] == "done":
            if ("ot" not in word.attrib.keys()):
                if word.attrib["wnsn"] != "0":
                    term = word.text
                    lemma = word.attrib["lemma"]
                    lexsn = word.attrib["lexsn"]
                    ss = sc2ss(lemma + '%' + lexsn)
                    if ss is not None and lemma not in stop_words:
                        result["term"] = term
                        result["lemma"] = lemma
                        result["sense"] = ss
                        found = True
        iterations += 1
    
    return result             

# %%
def get_context(sentence):
    words = set(sentence)
    words = set(map(lambda x:x.lower(),words))
    words = words.difference(stop_words)

    return words

def get_signature(sense):
    gloss = sense.definition()
    gloss = re.sub(r'[^\w\s]', '', gloss)
    gloss_set = set(gloss.split())

    examples = sense.examples()
    examples_set = set()
    for ex in examples:
        ex = re.sub(r'[^\w\s]', '', ex)
        examples_set.update(ex.split())
    signature = gloss_set.union(examples_set)
    signature = set(map(lambda x:x.lower(),signature))
    signature = signature.difference(stop_words)

    return signature

def compute_overlap(signature, context):
    return len(signature.intersection(context))

# %%
def Lesk(word, sentence):
    best_sense = wn.synsets(word)[0] # heuristic
    max_overlap = 0
    context = get_context(sentence)
    for ss in wn.synsets(word):
        signature = get_signature(ss)
        overlap = compute_overlap(signature, context)
        for hyp in ss.hyponyms():
            signature = get_signature(hyp)
            overlap += compute_overlap(signature, context)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = ss
    return best_sense

# %%
def get_possible_pair():
    found = False
    while not found:
        sentence = get_random_sentence()
        word = get_random_word(sentence)
        if word:
            found = True
    return sentence, word

# %%
num_corrects = 0
num_iterations = 10
num_phrases = 50
for i in range(num_iterations):
    for j in range(num_phrases):
        sentence, word = get_possible_pair()
        phrase = []
        for sent_word in sentence:
            phrase.append(sent_word.text)
        lemma = word["lemma"]
        best = Lesk(lemma, phrase)

        if best == word["sense"]:
            num_corrects += 1

print(num_corrects)
print(num_corrects/(num_iterations*num_phrases))


# %%
