import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn

frame_set = {
	1599: "Proliferating_in_number",
    320: "Try_defendant",
	195: "Political_locales",
	1609: "Sent_items",
	5: "Causation",
	49: "Emotion_heat",
	2197: "Losing",
	520: "Deny_or_grant_permission",
	1566: "Going_back_on_a_commitment",
	1253: "Referring_by_name"
}


# clean lists from the stopwords, punctuations; return a list of cleaned words
def clean_words(w_list):
    stopwords = nltk.corpus.stopwords.words('english')
    cleaned_row = []

    for w in w_list:
        if w not in stopwords and w.isalpha():
            cleaned_row.append(w)

    return cleaned_row


# retrieves the frames in framenet
def get_frames_dict():
    frame_w = {}
    frame_w_def = {}

    for f_id in frame_set.keys():
        f = fn.frame(f_id)
        frame_w[f_id] = {"name": frame_set[f_id], "fe": list(f.FE.keys()), "lu": list(f.lexUnit.keys())}
        frame_w_def[f_id] = {"name": f.definition, "fe": [f.FE[fe].definition for fe in f.FE.keys()],
                             "lu": [f.lexUnit[lu].definition for lu in f.lexUnit.keys()]}

    return frame_w, frame_w_def


# create context for s examples
def create_ex_s_context(s):
    ex_s = []

    for h in s.hypernyms():
        ex_s += h.definition().split()
        for ex in h.examples():
            ex_s += ex.split()

    for h in s.hyponyms():
        ex_s += h.definition().split()
        for ex in h.examples():
            ex_s += ex.split()

    return ex_s


# choose syns with bag of words approach
def assign_syns(name, set_w):
    correct_name = {"Proliferating_in_number": "proliferating",
                    "Try_defendant": "defendant",
                    "Political_locales": "politician",
                    "Sent_items": "sent",
                    "Causation": "causation",
                    "Emotion_heat": "emotion",
                    "Losing": "losing",
                    "Deny_or_grant_permission": "permission",
                    "Going_back_on_a_commitment": "commitment",
                    "Referring_by_name": "referring"
                    }

    n = correct_name[name]
    maximum = 0
    max_syns = None

    for s in wn.synsets(n):
        s_context = s.definition().lower().split()
        clean_s_cont = clean_words(s_context)

        ex_s_cont = create_ex_s_context(s)
        clean_ex_cont = clean_words(ex_s_cont)
        clean_s_cont += clean_ex_cont

        set_s = set(clean_s_cont)
        inters = set_w.intersection(set_s)

        if maximum < len(inters) + 1:
            maximum = len(inters) + 1
            max_syns = s

    return max_syns


# assign the correct syns to the fe
def assign_syns_fe(w, set_w):
    maximum = 0
    max_syns = None

    for s in wn.synsets(w):
        s_context = s.definition().lower().split()
        clean_s_cont = clean_words(s_context)

        ex_s_cont = create_ex_s_context(s)
        clean_ex_cont = clean_words(ex_s_cont)
        clean_s_cont += clean_ex_cont

        set_s = set(clean_s_cont)
        inters = set_w.intersection(set_s)

        if maximum < len(inters) + 1:
            maximum = len(inters) + 1
            max_syns = s

    return max_syns


# assign the correct syns to the lu
def assign_syns_lu(word, set_w, pos):
    maximum = 0
    max_syns = None

    for s in wn.synsets(word):
        if s.pos() != pos:
            continue

        s_context = s.definition().lower().split()
        clean_s_cont = clean_words(s_context)

        ex_s_cont = create_ex_s_context(s)
        clean_ex_cont = clean_words(ex_s_cont)
        clean_s_cont += clean_ex_cont

        set_s = set(clean_s_cont)
        inters = set_w.intersection(set_s)

        if maximum < len(inters) + 1:
            maximum = len(inters) + 1
            max_syns = s

    return max_syns


# disambiguate names, fes, lus
def disambiguate(f_name, f_def):
    words, syns = [], []

    for f_id in frame_set:
        name = frame_set[f_id]
        w_name = f_name[f_id]

        w_context = f_def[f_id]["name"].lower().split()
        clean_w_cont = clean_words(w_context)

        words.append(name)
        syns.append(assign_syns(name, set(clean_w_cont)))

        for w, w_cont in zip(w_name["fe"], f_def[f_id]["fe"]):

            w_cont = w_cont.lower().split()
            clean_context = clean_words(w_cont)

            syn = assign_syns_fe(w, set(clean_context))

            if syn:
                words.append(w.lower())
                syns.append(syn)

        for w, w_cont in zip(w_name["lu"], f_def[f_id]["lu"]):
            word, pos = w.split('.')
            w_cont = w_cont.lower().split()
            clean_context = clean_words(w_cont)

            syn = assign_syns_lu(word, set(clean_context), pos)

            if syn:
                words.append(w.lower())
                syns.append(syn)

    return words, syns


# take the manual annotations
def get_manual_annotation(annotation_path):
    annotations = []
    with open(annotation_path, encoding="utf8") as file:
        for row in enumerate(file):
            annotations.append(row[1])

    return annotations


# start the execution
def start():
    annotation_path = "gold_annotation.txt"
    out_path = "out/res.txt"
    f_name, f_def = get_frames_dict()
    words, syns = disambiguate(f_name, f_def)
    annotations = get_manual_annotation(annotation_path)

    count = 0
    with open(out_path, encoding="utf-8", mode='w') as new_file:

        correct_name = {"Proliferating": "proliferating",
                        "Try_defendant": "defendant",
                        "Political_locales": "politician",
                        "Sent_items": "sent",
                        "Causation": "causation",
                        "Emotion_heat": "emotion",
                        "Losing": "losing",
                        "Deny_or_grant_permission": "permission",
                        "Going_back_on_a_commitment": "commitment",
                        "Referring_by_name": "referring"
                        }

        for s, a in zip(syns, annotations):
            row = ""
            sy = s.name()
            word = str(a).split()[0]
            ann = str(a).split()[1]

            if sy == ann:
                count += 1

            if word[0].isupper():
                row += "\n" + "The word " + word + "was mapped with " + correct_name[word] + "\n"

            row += word + " " + ann + " " + sy + "\n"
            new_file.write(row)

        accuracy = count / len(syns)
        row = "\nThe overall accuracy is: "
        row += str(accuracy)
        new_file.write(row)
        print(accuracy)


start()
