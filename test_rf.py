import nltk
import sys
import pprint
import pickle
import numpy
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

data_path = 'data/'

print datetime.now()

fin1 = open(data_path + 'all_30_features_ranked_by_rf', 'r')
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

print datetime.now()
print "start to train the regression forest"
clf = RandomForestRegressor(n_estimators=1024, max_features=6)
clf.fit(tf_idf, useful_vote)
print "after training"

print datetime.now()
print "loading test samples"
fin3 = open(data_path + 'kaggle_test/test_set_all_30_features_ranked_by_rf', 'r')
test = pickle.load(fin3)
fin3.close()
print len(test)
print len(test[0])

print datetime.now()
print "start to do testing"
y_pred = clf.predict(test)
print "finish testing"
print datetime.now()

fout = open(data_path + 'kaggle_test/test_result_rf_1024_6_use_rf_ranked_feature', 'w')
pickle.dump(y_pred, fout)
fout.close()
