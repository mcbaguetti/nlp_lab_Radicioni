import csv
import similarity

file_path = "../esercitazione1/utils/WordSim353.csv"
wu_ans = []
sh_ans = []
lc_ans = []
human_meaning = []


with open(file_path) as file:
    reader = csv.reader(file)

    for row in reader:
        if row == 1:
            continue

        ans1, ans2, ans3 = similarity.word_to_synset(row[0], row[1])
        wu_ans.append(ans1)
        sh_ans.append(ans2)
        lc_ans.append(ans3)
        human_meaning.append(row[2])

