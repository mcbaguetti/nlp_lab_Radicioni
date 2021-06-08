# %%
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
import re
import os
import random

random.seed(1997) # TODO: togliere
# TODO: sostituire con lettura file
stop_words = ["a","able","about","above","abst","accordance","according","accordingly","across","act","actually","added","adj","adopted","affected","affecting","affects","after","afterwards","again","against","ah","all","almost","alone","along","already","also","although","always","am","among","amongst","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","are","aren","arent","arise","around","as","aside","ask","asking","at","auth","available","away","awfully","b","back","be","became","because","become","becomes","becoming","been","before","beforehand","begin","beginning","beginnings","begins","behind","being","believe","below","beside","besides","between","beyond","biol","both","brief","briefly","but","by","c","ca","came","can","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","could","couldnt","d","date","did","didn't","different","do","does","doesn't","doing","done","don't","down","downwards","due","during","e","each","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","et-al","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","few","ff","fifth","first","five","fix","followed","following","follows","for","former","formerly","forth","found","four","from","further","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","had","happens","hardly","has","hasn't","have","haven't","having","he","hed","hence","her","here","hereafter","hereby","herein","heres","hereupon","hers","herself","hes","hi","hid","him","himself","his","hither","home","how","howbeit","however","hundred","i","id","ie","if","i'll","im","immediate","immediately","importance","important","in","inc","indeed","index","information","instead","into","invention","inward","is","isn't","it","itd","it'll","its","itself","i've","j","just","k","keep","keeps","kept","keys","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","m","made","mainly","make","makes","many","may","maybe","me","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","more","moreover","most","mostly","mr","mrs","much","mug","must","my","myself","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","no","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","now","nowhere","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","omitted","on","once","one","ones","only","onto","or","ord","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","s","said","same","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","she","shed","she'll","shes","should","shouldn't","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","so","some","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","state","states","still","stop","strongly","sub","substantially","successfully","such","sufficiently","suggest","sup","sure","t","take","taken","taking","tell","tends","th","than","thank","thanks","thanx","that","that'll","thats","that've","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","these","they","theyd","they'll","theyre","they've","think","this","those","thou","though","thoughh","thousand","throug","through","throughout","thru","thus","til","tip","to","together","too","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","under","unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","very","via","viz","vol","vols","vs","w","want","wants","was","wasn't","way","we","wed","welcome","we'll","went","were","weren't","we've","what","whatever","what'll","whats","when","whence","whenever","where","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","while","whim","whither","who","whod","whoever","whole","who'll","whom","whomever","whos","whose","why","widely","willing","wish","with","within","without","won't","words","world","would","wouldn't","www","x","y","yes","yet","you","youd","you'll","your","youre","yours","yourself","yourselves","you've","z","zero"]
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
    abs_path = '/home/xps/nltk_data/corpora/semcor/' + local_path

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
    best_sense = wn.synsets(word)[0] # CHECK
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
