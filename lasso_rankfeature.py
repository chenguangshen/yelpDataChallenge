import nltk
import sys
import pprint
import pickle
import numpy
from sklearn import linear_model
from math import log, sqrt
from sklearn import cross_validation
from datetime import datetime
from sklearn.feature_selection import RFE

data_path = 'data/'


print datetime.now()

fin1 = open(data_path + 'tf_idf_all', 'r')
tf_idf = pickle.load(fin1)
fin1.close()
print len(tf_idf)
print len(tf_idf[0])

fin2 = open(data_path + 'useful_count', 'r')
useful_vote = pickle.load(fin2)
fin2.close()
print len(useful_vote)

tf_idf = numpy.array(tf_idf)
useful_vote = numpy.array(useful_vote)
# print len(tf_idf)
# for e in tf_idf:
# 	print len(e)

#sys.exit(0)

X = tf_idf
y = useful_vote
                
# Create the RFE object and rank each pixel
lasso = linear_model.Lasso(alpha=0.1)
rfe = RFE(estimator=lasso, n_features_to_select=20, step=0.05)
rfe.fit(X, y)
ranking = rfe.ranking_

print ranking

fin3 = open(data_path + 'var_importance_lasso', 'w')
pickle.dump(ranking, fin3)
fin3.close()
