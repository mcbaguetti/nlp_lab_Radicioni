# To add a new cell, type '# %%'
# %%
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn
import re
# %%
frame_set = {
    1599:"Proliferating_in_number",
	320:"Try_defendant",
	195:"Political_locales",
	1609:"Sent_items",
	5:"Causation"
}
for key, frame in frame_set.items():
    print(key,frame)
# %%
