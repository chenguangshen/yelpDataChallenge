import nltk
import numpy
import sys
import re
import pprint
import pickle
from nltk.corpus import stopwords
from nltk import PorterStemmer
import unicodedata
from datetime import datetime

data_path = "E:\workspace\cs246_data\\"

fi1 = open(data_path + 'unique_terms', 'r')
unique_terms = pickle.load(fi1)
fi1.close()
print "No. of unique terms: ", len(unique_terms)

sorted_index = [209, 102, 42, 356, 263, 194, 75, 81, 375, 364,
 353, 314, 96, 216, 26, 56, 289, 174, 197, 378]

words = [unique_terms[i] for i in sorted_index]

pprint.pprint(words)

# ['later', 'sat', 'brought', 'bite', 'arriv', 'saw', 'share', 'turn', 'felt', 'instead', 'head', 'read', 'tomato', 'fact', 'abl', 'type', 'complet', 'guess', 'mix', 'ago']