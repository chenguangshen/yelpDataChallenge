import nltk
import sys
import pprint
import pickle
import numpy
from sklearn.svm import SVR
from math import log, sqrt
from sklearn import cross_validation
from datetime import datetime

def rmsle(x, y):
	res = 0
	if (len(x) == len(y)):
		for i in range(len(x)):
			res = res + (log(y[i] + 1) - log(x[i] + 1)) * (log(y[i] + 1) - log(x[i] + 1))
		return sqrt(res / len(x))
	else:
		print "dimension not equal!"
		return -1

data_path = 'data/'

print datetime.now()

print "feature ranked by tfidf"
fin1 = open(data_path + 'all_30_features_ranked_by_tfidf', 'r')
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

kf = cross_validation.KFold(len(tf_idf), n_folds=5)

# pprint.pprint(tf_idf)

total_rmsle = 0
count = 0
print datetime.now()
print "begin to do regression, vote not log scaled, SVR with rbf kernel g=0.015, len=", len(tf_idf)
for train_index, test_index in kf:
	count = count + 1
	print "5-fold CV No.",  + count
	# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = tf_idf[train_index], tf_idf[test_index]
	y_train, y_test = useful_vote[train_index], useful_vote[test_index]
	clf = SVR(kernel='rbf', gamma=0.015)
	clf.fit(X_train, y_train)
	y_score = clf.predict(X_test)
	res = rmsle(y_test, y_score)
	print "rmsle=", res
	total_rmsle = total_rmsle + res
print "finish doing regression"
print "5-fold CV avg rmsle=", (total_rmsle / 5.0)
print datetime.now()
#pprint.pprint(clf.predict([[0]*300 + [0.1]*85, [0]*300 + [0.3]*85]))
