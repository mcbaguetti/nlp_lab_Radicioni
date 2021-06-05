# %%
import nltk
from nltk.corpus import wordnet as wn

# %%
ss = wn.synset("cat.n.01")
for hyper in ss.hypernyms(): 
    print(ss)
    print(f'hyper: {str(hyper)}')
    print(hyper.name())
    ss = wn.synset(hyper.name())

# %%
s1 = wn.synset("cat.n.01")
s2 = wn.synset("tiger.n.02")
s3 = wn.synset("entity.n.01")

print(s1.min_depth())
print(s3.min_depth())

print(s1.shortest_path_distance(s2))
print(s1.lowest_common_hypernyms(s2))
print(s1.lowest_common_hypernyms(s2)[0].min_depth())


# %%
for ss in wn.synsets("tiger"):
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
