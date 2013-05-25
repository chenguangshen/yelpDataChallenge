import nltk
import sys
import pprint
import pickle
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from datetime import datetime

# rf best result:
# 1024 tree, 6 feature, 20 text feature ranked by rf + 10 non text features
# score: 0.51689

data_path = 'data/'

print datetime.now()

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

print datetime.now()
#print "start to train the regression forest"
#clf = RandomForestRegressor(n_estimators=1024, max_features=6)
print "start to train the lasso"
clf = clf = linear_model.Lasso(alpha=1e-5)
clf.fit(tf_idf, useful_vote)
print "after training"

print datetime.now()
print "loading test samples"
fin3 = open(data_path + 'kaggle_test/test_set_all_30_features_ranked_by_lasso', 'r')
test = pickle.load(fin3)
fin3.close()
print len(test)
print len(test[0])

print datetime.now()
print "start to do testing"
y_pred = clf.predict(test)
print "finish testing"
print datetime.now()

fout = open(data_path + 'kaggle_test/test_result_rf_1024_6_use_lasso_ranked_feature', 'w')
pickle.dump(y_pred, fout)
fout.close()
