import nltk
import sys
import pprint
import pickle
import numpy
from sklearn.svm import SVR
data_path = 'E:\workspace\cs246_data\\'

fin1 = open(data_path + 'SVR_words_only', 'r')
clf = pickle.load(fin1)
fin1.close()

clf.predict([0]*300 + [0.1]*85)

# # for e in tf_idf:
# # 	print len(e)

# pprint.pprint(tf_idf[0])
# print len(tf_idf[0])