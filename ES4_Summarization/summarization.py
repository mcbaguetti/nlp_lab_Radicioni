from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# 0 reduce docs by 10, 20, 30 %
# 1 find topic like a nasari vectors (use title or first paragraph), use wsd
# 2 create context, group vectors of step 1
# 3 use weightened overlap and other methods for saving and reranking paragraphs
# 4 eval with blue and rouge


def create_nasari():

    nasari = {}
    nasari_path = "utils/NASARI_vectors/dd-small-nasari-15.txt"

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


def find_topic(file_path, nasari):

    topic = []
    count_words = 10  # num of words needed for finding a topic

    with open(file_path, encoding="utf8") as file:
        for row in file:
            new_row = row.strip()
            if not new_row:
                continue

            final_row = clean_row(new_row)

            for word in final_row:
                new_word = word.lower()
                if new_word in nasari:
                    count_words -= 1
                    topic.append(nasari[new_word])

                if count_words == 0:
                    return topic


def start(file_path):

    nasari_dict = create_nasari()
    topic_vect = find_topic(file_path, nasari_dict)
    print(topic_vect)


start("utils/docs/Andy-Warhol.txt")
