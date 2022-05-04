import math

from nltk.tokenize import sent_tokenize, word_tokenize
import nltk


# 0 reduce docs by 10, 20, 30 %
# 1 find topic like a nasari vectors (use title or first paragraph), use wsd
# 2 create context, group vectors of step 1
# 3 use weightened overlap and other methods for saving and reranking paragraphs
# 4 eval with blue and rouge


# create a dictionary with pairs word: nasari vector
def create_nasari():
    nasari = {}
    nasari_path = "NASARI_vectors/dd-small-nasari-15.txt"

    with open(nasari_path, encoding="utf8") as file:
        for row in file:
            vector = []
            new_row = row.split(";")
            term = new_row[1].lower()

            for elem in new_row[2:]:
                # handle the \n at the end of the row
                if elem == new_row[len(new_row) - 1]:
                    elem.strip()

                # find word and associated score
                splitted = elem.split("_")
                if len(splitted) == 2:
                    vector.append((splitted[0].lower(), float(splitted[1])))

            nasari[term] = vector
        return nasari


# clean every row from the stopwords, punctuations and lower the case; return a list of cleaned words
def clean_row(row):
    stopwords = nltk.corpus.stopwords.words('english')
    cleaned_row = []
    sentence_tokenized = sent_tokenize(row.lower())

    for sentence in sentence_tokenized:
        word_tokens = word_tokenize(str(sentence))

        for w in word_tokens:
            if w not in stopwords and w.isalpha():
                cleaned_row.append(w)

    return cleaned_row


# take the title for creating a topic; if the title produces less than 10 nasari vectors check the next row
def find_title_topic(file_path, nasari):
    topic = []
    topic_words = 10  # min num of words needed for finding a topic
    count_row = 0

    with open(file_path, encoding="utf8") as file:
        for row in file:
            count_row += 1
            final_row = clean_row(row)

            for word in final_row:
                if word.lower() in nasari:
                    topic_words -= 1
                    topic.append(nasari[word.lower()])

                if count_row != 0 and topic_words <= 0:
                    return topic


# use the cue method for creating a topic; at max the topic will contains 10 nasari vectors
def find_cue_topic(file_path, nasari):
    score_paragraph = {}  # dict for ranking paragraph => paragraph_idx : rank
    topic = []

    bonus_words = ['better', 'worse', 'less', 'more', 'further', 'farther', 'best', 'worst', 'least', 'most',
                   'furthest', 'farthest', 'more', 'important', 'seen', 'all', 'fact', 'final', 'analysis',
                   'whole', 'brief', 'altogether', 'obviously', 'overall', 'ultimately', 'ordinarily',
                   'definitely', 'usually', 'emphasize', 'result', 'henceforth', 'additionally', 'main',
                   'aim', 'purpose', 'outline', 'investigation']
    stigma_words = ['no', 'not', 'i', 'you', 'she', 'he', 'we', 'they', 'it', 'me', 'him', 'her', 'us', 'them', 'mine',
                    'ours', 'hers', 'theirs', 'ourselves', 'myself', 'himself', 'who', 'whose', 'which', 'what', 'this',
                    'that', 'these', 'those', 'whom', 'whose']

    with open(file_path, encoding="utf8") as file:
        for i, row in enumerate(file):
            score = 0
            for word in row:
                if word in bonus_words:
                    score += 1
                if word in stigma_words:
                    score -= 1

            score_paragraph[i] = score

    best_paragraph_idx = max(score_paragraph, key=score_paragraph.get)

    with open(file_path, encoding="utf8") as file:
        for i, row in enumerate(file):
            if i != best_paragraph_idx:
                continue

            final_row = clean_row(row)

            for word in final_row:
                if word.lower() in nasari:
                    topic.append(nasari[word.lower()])

    return topic


# calculate the rank of q dimension and v vector
def rank(q, v):
    for i, w in enumerate(v):
        if w[0] == q:
            return i + 1


# find the weightened overlap between two vectors
def calc_overlap(v1, v2):
    wo = 0
    word_idx = 0
    common_words = []

    for w1 in v1:
        for w2 in v2:
            if w1[word_idx] == w2[word_idx]:
                common_words.append(w1[word_idx])

    if common_words:
        n = 0
        d = 1

        for q in common_words:
            n += 1 / (rank(q, v1) + rank(q, v2))

        for i in range(1, len(common_words) + 1):
            d += 1 / (2 * i)

        wo = n / d

    return wo


# calculate the semantic similarity between the topic vector and the paragraph vector
def calc_sim(topic_vect, paragraph_vector):
    sim = 0

    for vector1 in paragraph_vector:
        for vector2 in topic_vect:
            sim = max(math.sqrt(calc_overlap(vector1, vector2)), sim)

    return sim


# process the txt for each row, find the similarity between topic and paragraph
def process_doc(file_path, topic_vect, nasari_dict, max_lines):
    paragraph_val = {}  # row number is the key, the score of WO is the value

    with open(file_path, encoding="utf8") as file:
        for i, row in enumerate(file):
            # skip the title
            if i == 0:
                continue
            # check if we scanned all the reduced doc
            if i > max_lines:
                return paragraph_val

            paragraph_vector = []
            final_row = clean_row(row)

            for word in final_row:
                if word.lower() in nasari_dict:
                    paragraph_vector.append(nasari_dict[word.lower()])

            paragraph_score = calc_sim(topic_vect, paragraph_vector)
            paragraph_val[i] = paragraph_score

    return paragraph_val


# calculate the rows to process in the document
def calc_rows_to_read(file_path, percentage):
    with open(file_path, encoding="utf8") as file:
        return int(len(file.readlines()) * percentage)


# start the summarization process
def start(file_path, percentage, topic_method):
    topic_vect = []

    nasari_dict = create_nasari()
    lines = calc_rows_to_read(file_path, percentage)

    if topic_method == "title":
        topic_vect = find_title_topic(file_path, nasari_dict)
    elif topic_method == "cue":
        topic_vect = find_cue_topic(file_path, nasari_dict)

    paragraph_dict = process_doc(file_path, topic_vect, nasari_dict, lines)
    print(paragraph_dict)


start("docs/Andy-Warhol.txt", 0.9, "title")
