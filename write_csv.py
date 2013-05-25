import nltk
import sys
import json
import pprint
import pickle
import numpy
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

data_path = 'data/'

fin1 = open(data_path + 'kaggle_test/test_result_rf_1024_6_use_rf_ranked_feature', 'r')
pred = pickle.load(fin1)
fin1.close()
print len(pred)

fout = open(data_path + "submission/submission_rf_1024_6_ranked_by_rf.csv", "w")
print >>fout, "id,votes" 
f = open("/media/Data/workspace/Dataset/yelp_kaggle/yelp_test_set/yelp_test_set_review.json")
for i in range(22956):
	r = unicode(f.readline())
	d = json.loads(r)
	rid = d['review_id']
	print >>fout, rid + "," + str(pred[i])
f.close()
fout.close()
