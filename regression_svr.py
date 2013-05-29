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
fout = open(data_path + 'svr_trial_result', 'w')

print datetime.now()

fin1 = open(data_path + 'all_30_features_ranked_by_tfidf', 'r')
tf_idf = pickle.load(fin1)
fin1.close()
print len(tf_idf)
print >>fout, len(tf_idf)
print len(tf_idf[0])
print >>fout, len(tf_idf[0])

fin2 = open(data_path + 'useful_count', 'r')
useful_vote = pickle.load(fin2)
fin2.close()
print len(useful_vote)
print >>fout, len(useful_vote)


tf_idf = numpy.array(tf_idf)
useful_vote = numpy.array(useful_vote)
# print len(tf_idf)
# for e in tf_idf
# 	print len(e)

kf = cross_validation.KFold(len(tf_idf), n_folds=5)

# pprint.pprint(tf_idf)
rad = [0.5,1,2]
deg = [2]

best_score = 10000
best_kernel = None

print "try linear kernel"
print >>fout, "try linear kernel"
total_rmsle = 0
count = 0
print datetime.now()
print "begin to do regression, vote not log scaled, SVR with linear kernel, len=", len(tf_idf)
print >>fout, "begin to do regression, vote not log scaled, SVR with linear kernel, len=", len(tf_idf)
kernel = "linear"
for train_index, test_index in kf:
	count = count + 1
	print "5-fold CV No.",  + count
	print >>fout,"5-fold CV No.",  + count
	# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = tf_idf[train_index], tf_idf[test_index]
	y_train, y_test = useful_vote[train_index], useful_vote[test_index]
	clf = SVR(kernel='linear')
	clf.fit(X_train, y_train)
	y_score = clf.predict(X_test)
	res = rmsle(y_test, y_score)
	print "rmsle=", res
	print >>fout,"rmsle=", res
	total_rmsle = total_rmsle + res
	if count == 1:
		break
print "finish doing regression"
print >>fout, "finish doing regression"
score = total_rmsle / 5.0
print "5-fold CV avg rmsle=", score
print >>fout, "5-fold CV avg rmsle=", score
print datetime.now()
if score < best_score:
	best_score = score
	best_kernel = kernel

print "try poly kernels"
print >>fout, "try poly kernels"
for d in deg:
	total_rmsle = 0
	count = 0
	print datetime.now()
	kernel = "poly kernel degree=", d
	print "begin to do regression, vote not log scaled, SVR with", kernel, " len=", len(tf_idf)
	print >>fout, "begin to do regression, vote not log scaled, SVR with", kernel, " len=", len(tf_idf)
	for train_index, test_index in kf:
        	count = count + 1
       		print "5-fold CV No.",  + count
       		print >>fout, "5-fold CV No.",  + count
        	X_train, X_test = tf_idf[train_index], tf_idf[test_index]
        	y_train, y_test = useful_vote[train_index], useful_vote[test_index]
        	clf = SVR(kernel='poly', degree=d)
        	clf.fit(X_train, y_train)
        	y_score = clf.predict(X_test)
       		res = rmsle(y_test, y_score)
        	print "rmsle=", res
        	print >>fout, "rmsle=", res
        	total_rmsle = total_rmsle + res
		if count == 1:
			break;
	print "finish doing regression"
	print >>fout, "finish doing regression"
	score = total_rmsle / 5.0
	print "5-fold CV avg rmsle=", score
	print >>fout, "5-fold CV avg rmsle=", score
	print datetime.now()
	if score < best_score:
        	best_score = score
       		best_kernel = kernel

print "try rbf kernels"
print >>fout, "try rbf kernels"
for r in rad:
        total_rmsle = 0
        count = 0
        print datetime.now()
        kernel = "rbf kernel gamma=", r
        print "begin to do regression, vote not log scaled, SVR with", kernel, " len=", len(tf_idf)
        print >>fout,"begin to do regression, vote not log scaled, SVR with", kernel, " len=", len(tf_idf)
        for train_index, test_index in kf:
                count = count + 1
                print "5-fold CV No.",  + count
                print >>fout,"5-fold CV No.",  + count
                X_train, X_test = tf_idf[train_index], tf_idf[test_index]
                y_train, y_test = useful_vote[train_index], useful_vote[test_index]
                clf = SVR(kernel='rbf', gamma=r)
                clf.fit(X_train, y_train)
                y_score = clf.predict(X_test)
                res = rmsle(y_test, y_score)
                print "rmsle=", res
                print >>fout, "rmsle=", res
                total_rmsle = total_rmsle + res
		if count == 1:
			break;
        print "finish doing regression"
        print >>fout, "finish doing regression"
        score = total_rmsle / 5.0
        print "5-fold CV avg rmsle=", score
        print >>fout, "5-fold CV avg rmsle=", score
        print datetime.now()
        if score < best_score:
                best_score = score
                best_kernel = kernel

print "best_score=", best_score
print >>fout, "best_score=", best_score
print "best_kernel=", best_kernel
print >>fout, "best_kernel=", best_kernel
fout.close()
#pprint.pprint(clf.predict([[0]*300 + [0.1]*85, [0]*300 + [0.3]*85]))
