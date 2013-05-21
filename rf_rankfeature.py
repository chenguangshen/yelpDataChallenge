import nltk
import sys
import pprint
import pickle
import numpy
from sklearn.ensemble import RandomForestRegressor
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

total_rmsle = 0
count = 0

print datetime.now()
print "begin to do regression, vote not log scaled, RF with 32 trees, len=", len(tf_idf)
X_train, X_test = tf_idf[0:220000], tf_idf[220000:]
y_train, y_test = useful_vote[0:220000], useful_vote[220000:]
clf = RandomForestRegressor(n_estimators=16, max_features=6, compute_importances=True)
clf.fit(X_train, y_train)
y_score = clf.predict(X_test)
res = rmsle(y_test, y_score)
print "rmsle=", res
print datetime.now()

imt = numpy.array(clf.feature_importances_)
print imt

fin3 = open(data_path + 'var_importance_rf', 'w')
pickle.dump(imt, fin3)
fin3.close()
