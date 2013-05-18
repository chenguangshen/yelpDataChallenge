import nltk
import sys
import pprint
import pickle
import numpy
from sklearn.ensemble import RandomForestRegressor
from math import log, sqrt, exp
from sklearn import cross_validation
from datetime import datetime

def rmsle(x, y):
	res = 0
	if (len(x) == len(y)):
		for i in range(len(x)):
			res = res + (log(exp(y[i]) - 0.01 + 1) - log(exp(x[i]) - 0.01 + 1)) * (log(exp(y[i]) - 0.01 + 1) - log(exp(x[i]) - 0.01 + 1))
		return sqrt(res / len(x))
	else:
		print "dimension not equal!"
		return -1

data_path = 'E:\workspace\cs246_data\\'

print datetime.now()

fin1 = open(data_path + 'all_30_features', 'r')
tf_idf = pickle.load(fin1)
fin1.close()
print len(tf_idf)
print len(tf_idf[0])

fin2 = open(data_path + 'useful_count', 'r')
useful_vote = pickle.load(fin2)
fin2.close()
print len(useful_vote)

tf_idf = numpy.array(tf_idf)
useful_vote = [log(w + 0.01) for w in useful_vote]
useful_vote = numpy.array(useful_vote)
# print len(tf_idf)
# for e in tf_idf:
# 	print len(e)

kf = cross_validation.KFold(len(tf_idf), n_folds=5)

# pprint.pprint(tf_idf)

total_rmsle = 0
count = 0
print datetime.now()
print "begin to do regression, vote log scaled, RF with 32 trees, len=", len(tf_idf)
for train_index, test_index in kf:
	count = count + 1
	print "5-fold CV No.",  + count
	# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = tf_idf[train_index], tf_idf[test_index]
	y_train, y_test = useful_vote[train_index], useful_vote[test_index]
	clf = RandomForestRegressor(n_estimators=32)
	clf.fit(X_train, y_train)
	y_score = clf.predict(X_test)
	res = rmsle(y_test, y_score)
	print "rmsle=", res
	total_rmsle = total_rmsle + res
print "finish doing regression"
print "5-fold CV avg rmsle=", (total_rmsle / 5.0)
print datetime.now()
#pprint.pprint(clf.predict([[0]*300 + [0.1]*85, [0]*300 + [0.3]*85]))