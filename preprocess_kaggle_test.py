import json
import nltk
import numpy
import sys
import re
import pprint
import pickle
from nltk.corpus import stopwords
from nltk import PorterStemmer
import unicodedata
from datetime import datetime

def normalize(x):
	min = 1e10
	max = 0.0
	for e in x:
		if e > max:
			max = e
		if e < min:
			min = e
	return [float(w - min) / float(max - min) for w in x]

data_path = "data/"
f = open("/media/Data/workspace/Dataset/yelp_kaggle/yelp_test_set/yelp_test_set_review.json")
#f = open("/Users/cgshen/Dataset/yelp_test_set/yelp_test_set_review.json")
fuser = open("/media/Data/workspace/Dataset/yelp_phoenix_academic_dataset/yelp_academic_dataset_user.json")

fi1 = open(data_path + 'unique_terms', 'r')
unique_terms = pickle.load(fi1)
fi1.close()
print "No. of unique terms: ", len(unique_terms)

users = {}
for i in range(43873):
	# if i % 100 == 0:
	# 	print i
	r = unicode(fuser.readline())
	d = json.loads(r)
	users[d['user_id']] = d
fuser.close()
print "after reading all users..."

docs = []
rates = []
dates = []
lens = []
today = datetime.strptime('2013-05-21', "%Y-%m-%d")
punctuation = re.compile(r'[-.?!,`"$\':;()|0-9]')
avg_rates = []
total_reviews = []
usefuls = []
cools = []
funnys = []

for i in range(22956):
	r = unicode(f.readline())
	d = json.loads(r)
	terms = nltk.Text(nltk.word_tokenize(unicodedata.normalize('NFKD', d["text"]).encode('ascii','ignore')))
	terms = [w.lower() for w in terms]
	terms = [punctuation.sub("", w) for w in terms]
	terms = [w for w in terms if not w in stopwords.words('english')]
	terms = filter(len, terms)
	for i in range(len(terms)):
		terms[i] = PorterStemmer().stem_word(terms[i])
	terms = [w for w in terms if w in unique_terms]
	docs.append(terms)

	rate = d['stars']
	rates.append(rate)
	doc = unicodedata.normalize('NFKD', d["text"]).encode('ascii','ignore')
	lens.append(len(doc))
	date = datetime.strptime(d['date'], "%Y-%m-%d")
	time = (today - date).days
	dates.append(time)

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

print "Prepared ", len(docs), " documents..."
print len(docs[0])

fin2 = open(data_path + 'kaggle_test/test_review_after_preprocess', 'w')
pickle.dump(docs, fin2)
fin2.close()

# # all terms: 8939526
# # unique terms: 385

fin3 = open(data_path + 'all_doc_without_uncommon', 'r')
all_docs = pickle.load(fin3)
fin3.close()
print len(all_docs)

collection = nltk.TextCollection(all_docs)

def TFIDF(document):
    word_tfidf = []
    for word in unique_terms:
        word_tfidf.append(collection.tf_idf(word, document))
    return word_tfidf

vectors = []
for f in docs:
	if len(f) == 0:
		vectors.append([0] * len(unique_terms))
	else:
		vectors.append(numpy.array(TFIDF(f)))
print "Vectors created."

#ranked by tfidf
#sorted_index = [209, 102, 42, 356, 263, 194, 75, 81, 375, 364,
# 353, 314, 96, 216, 26, 56, 289, 174, 197, 378]

# rankded by rf
#sorted_index = [61, 361, 281, 243,  357, 180, 363, 327, 160, 86,
# 321, 280, 224, 62, 271, 309, 347, 44, 95, 193]

#ranked by lasso
sorted_index = [2, 34, 37, 75, 119, 121, 123, 135, 157,
 191, 210, 216, 246, 263, 276, 316, 349, 353, 356, 379]

vectors = [[e[i] for i in sorted_index] for e in vectors]

review_rate = normalize(rates)
review_date = normalize(dates)
review_length = normalize(lens)
user_avg_rate = normalize(avg_rates)
user_total_review = normalize(total_reviews)
user_total_useful = normalize(usefuls)
user_total_cool = normalize(cools)
user_total_funny = normalize(funnys)

features = []
for i in range(22956):
	if i % 100 == 0:
		print i
	t = []
	t.extend(vectors[i])
	t.append(review_rate[i])
	t.append(review_length[i])
	t.append(review_date[i])
	t.append(0)
	t.append(0)
	t.append(user_avg_rate[i])
	t.append(user_total_review[i])
	t.append(user_total_useful[i])
	t.append(user_total_cool[i])
	t.append(user_total_funny[i])
	#print t
	features.append(t)

fout = open(data_path + 'kaggle_test/test_set_all_30_features_ranked_by_lasso', 'w')
pickle.dump(features, fout)
fout.close()
