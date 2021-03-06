import nltk
import sys
import pprint
import pickle
import numpy
from math import log, sqrt
from sklearn import linear_model
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
print "30 features ranked by lasso"
fin1 = open(data_path + 'all_30_features_ranked_by_lasso', 'r')
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


alpha = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
best_alpha = -1
best_score = 10000

for aaa in alpha:

	kf = cross_validation.KFold(len(tf_idf), n_folds=5)

	total_rmsle = 0
	count = 0

	print datetime.now()
	print "begin to do regression, vote not log scaled, lasso a=", aaa, " ,len=", len(tf_idf)
	print "features ranked by lasso"
	for train_index, test_index in kf:
		count = count + 1
		print "5-fold CV No.",  + count
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = tf_idf[train_index], tf_idf[test_index]
		y_train, y_test = useful_vote[train_index], useful_vote[test_index]
	        clf = linear_model.Lasso(alpha=aaa)
		clf.fit(X_train, y_train)
		y_score = clf.predict(X_test)
		res = rmsle(y_test, y_score)
		print "rmsle=", res
		total_rmsle = total_rmsle + res
	#print "finish doing regression"
	score = (total_rmsle / 5.0)
	print "5-fold CV avg rmsle=", score

	if score < best_score:
		best_score = score
		best_alpha = aaa

print datetime.now()
print best_score
print best_alpha
