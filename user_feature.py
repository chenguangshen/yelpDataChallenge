import json
import nltk
import sys
import pprint
import pickle
import numpy
import unicodedata
from datetime import datetime

data_path = 'data/user_features/'
f = open("/media/Data/workspace/Dataset/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json")
fuser = open("/media/Data/workspace/Dataset/yelp_phoenix_academic_dataset/yelp_academic_dataset_user.json")
#f = open("/Users/cgshen/Dataset/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json")
#fuser = open("/Users/cgshen/Dataset/yelp_phoenix_academic_dataset/yelp_academic_dataset_user.json")

def normalize(x):
	min = 1e10
	max = 0.0
	for e in x:
		if e > max:
			max = e
		if e < min:
			min = e
	return [float(w - min) / float(max - min) for w in x]
users = {}

for i in range(43873):
	if i % 100 == 0:
		print i
	r = unicode(fuser.readline())
	d = json.loads(r)
	users[d['user_id']] = d
fuser.close()

print "after reading all users..."

avg_rates = []
total_reviews = []
usefuls = []
cools = []
funnys = []

for i in range(229907):
	if i % 100 == 0:
		print i
	r = unicode(f.readline())
	d = json.loads(r)
	uid = d['user_id']
	
	if uid in users:
		u = users[uid]
		avg_rates.append(float(u['average_stars']))
		total_reviews.append(int(u['review_count']))
		usefuls.append(int(u['votes']['useful']))
		cools.append(int(u['votes']['cool']))
		funnys.append(int(u['votes']['funny']))
	else:
		#print "no corresponding user!"
		avg_rates.append(0)
		total_reviews.append(0)
		usefuls.append(0)
		cools.append(0)
		funnys.append(0)
f.close()

# pprint.pprint(avg_rates)
# pprint.pprint(normalize(avg_rates))
# pprint.pprint(total_reviews)
# pprint.pprint(normalize(total_reviews))
# pprint.pprint(usefuls)
# pprint.pprint(normalize(usefuls))
# pprint.pprint(cools)
# pprint.pprint(normalize(cools))
# pprint.pprint(funnys)
# pprint.pprint(normalize(funnys))

avg_rates = normalize(avg_rates)
total_reviews = normalize(total_reviews)
usefuls = normalize(usefuls)
cools = normalize(cools)
funnys = normalize(funnys)

print len(avg_rates)
print len(total_reviews)
print len(usefuls)
print len(cools)
print len(funnys)

fi1 = open(data_path + 'avg_rating_feature', 'w')
pickle.dump(avg_rates, fi1)
fi1.close()

fi2 = open(data_path + 'total_review_feature', 'w')
pickle.dump(total_reviews, fi2)
fi2.close()

fi3 = open(data_path + 'total_useful_feature', 'w')
pickle.dump(usefuls, fi3)
fi3.close()

fi4 = open(data_path + 'total_cool_feature', 'w')
pickle.dump(cools, fi4)
fi4.close()

fi5 = open(data_path + 'total_funny_feature', 'w')
pickle.dump(funnys, fi5)
fi5.close()