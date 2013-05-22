import json
import nltk
import sys
import pprint
import pickle
import numpy
import unicodedata
from datetime import datetime

data_path = 'data/nontext_features/'
#f = open("E:\workspace\Dataset\yelp_phoenix_academic_dataset\yelp_academic_dataset_review.json")
f = open("/media/Data/workspace/Dataset/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json")

def normalize(x):
	min = 1e10
	max = 0.0
	for e in x:
		if e > max:
			max = e
		if e < min:
			min = e
	return [float(w - min) / float(max - min) for w in x]

rates = []
dates = []
lens = []

cools = []
funnys = []

today = datetime.strptime('2013-05-21', "%Y-%m-%d")
for i in range(229907):
	if i % 100 == 0:
		print i
	r = unicode(f.readline())
	d = json.loads(r)
	funnys.append(int(d['votes']['funny']))
	cools.append(int(d['votes']['cool']))
	rate = d['stars']
	rates.append(rate)
	doc = unicodedata.normalize('NFKD', d["text"]).encode('ascii','ignore')
	lens.append(len(doc))
	date = datetime.strptime(d['date'], "%Y-%m-%d")
	time = (today - date).days
	dates.append(time)
f.close()

# pprint.pprint(cools)
# pprint.pprint(normalize(cools))
# pprint.pprint(funnys)
# pprint.pprint(normalize(funnys))

# pprint.pprint(rates)
# pprint.pprint(normalize(rates))
# pprint.pprint(dates)
# pprint.pprint(normalize(dates))
# pprint.pprint(lens)
# pprint.pprint(normalize(lens))

rates = normalize(rates)
dates = normalize(dates)
lens = normalize(lens)
cools = normalize(cools)
funnys = normalize(funnys)

fi1 = open(data_path + 'rating_feature', 'w')
pickle.dump(rates, fi1)
fi1.close()

fi2 = open(data_path + 'date_feature', 'w')
pickle.dump(dates, fi2)
fi2.close()

fi3 = open(data_path + 'length_feature', 'w')
pickle.dump(lens, fi3)
fi3.close()

fia1 = open(data_path + 'cool_feature', 'w')
pickle.dump(cools, fia1)
fia1.close()

fia2 = open(data_path + 'funny_feature', 'w')
pickle.dump(funnys, fia2)
fia2.close()
