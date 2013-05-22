import nltk
import numpy
import sys
import re
import pprint
import pickle

data_path = "data/"

fin1 = open(data_path + 'tf_idf_all', 'r')
tf_idf = pickle.load(fin1)
fin1.close()
print len(tf_idf)

sorted_index = [61, 361, 281, 243,  357, 180, 363, 327, 160, 86,
 321, 280, 224, 62, 271, 309, 347, 44, 95, 193]

tf_idf_top = [[e[i] for i in sorted_index] for e in tf_idf]

print len(tf_idf_top)
print len(tf_idf_top[0])

fin2 = open(data_path + 'tf_idf_top_20_rf', 'wb')
pickle.dump(tf_idf_top, fin2)
fin2.close()


# fin1 = open(data_path + 'tf_idf_top_20_rf', 'r')
# tf_idf = pickle.load(fin1)
# fin1.close()
# print len(tf_idf)
# print len(tf_idf[0])