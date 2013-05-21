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

data_path = "data/"

fi1 = open(data_path + 'unique_terms', 'r')
unique_terms = pickle.load(fi1)
fi1.close()
print "No. of unique terms: ", len(unique_terms)

# sort just according to tf-idf index
# sorted_index = [209, 102, 42, 356, 263, 194, 75, 81, 375, 364,
#  353, 314, 96, 216, 26, 56, 289, 174, 197, 378]

# sorted_index = [209, 102, 42, 356, 263, 194, 75, 81, 375, 364,
#  353, 314, 96, 216, 26, 56, 289, 174, 197, 378]


# sort using random forest's variable importance
sorted_index = [61, 361, 281, 243,  357, 180, 363, 327, 160, 86,
 321, 280, 224, 62, 271, 309, 347, 44, 95, 193]

words = [unique_terms[i] for i in sorted_index]

pprint.pprint(words)

# rank tfidf:
# ['later', 'sat', 'brought', 'bite', 'arriv', 'saw', 'share', 'turn', 'felt', 'instead', 'head', 'read', 'tomato', 'fact', 'abl', 'type', 'complet', 'guess', 'mix', 'ago']

# rf variable importance:
# ['appet',  'wish', 'sushi', 'husband', 'spici', 'especi', 'problem',
#  'often', 'waitress', 'disappoint', 'server', 'almost', 'absolut', 'impress',
#  'beer', 'els', 'custom', 'famili', 'salsa', 'breakfast']