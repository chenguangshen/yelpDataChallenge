import json
import nltk
import numpy
import sys
import re
import pprint
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import PorterStemmer
from nltk.book import FreqDist
import unicodedata


#f = open("E:\workspace\Dataset\yelp_phoenix_academic_dataset\yelp_academic_dataset_review.json")
#f = open("/Users/cgshen/Dataset/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json")
f = open("/media/Data/workspace/Dataset/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json")

docs = []
use_vote = []

for i in range(229907):
	r = unicode(f.readline())
	d = json.loads(r)
	doc = nltk.Text(nltk.word_tokenize(unicodedata.normalize('NFKD', d["text"]).encode('ascii','ignore')))
	docs.append(doc)
	useful = int(d["votes"]["useful"])
	use_vote.append(useful)
f.close()

print "Prepared ", len(docs), " documents..."

fi111 = open('data/raw_docs', 'w')
pickle.dump(docs, fi111)
fi111.close()

fi12 = open('data/useful_count', 'w')
pickle.dump(use_vote, fi12)
fi12.close()

punctuation = re.compile(r'[-.?!,`"$\':;()|0-9]')
fdist = FreqDist()

count = 0
all_doc = []
for doc in docs:
	count = count + 1
	collection = doc #nltk.TextCollection(doc)
	print "for document ", count, " created a collection of", len(collection), "terms."

	terms = list(collection)
	terms = [w.lower() for w in terms]
	terms = [punctuation.sub("", w) for w in terms]
	terms = [w for w in terms if not w in stopwords.words('english')]
	terms = filter(len, terms)
	#print "terms found after removing stop words: ", len(terms)

	for i in range(len(terms)):
		terms[i] = PorterStemmer().stem_word(terms[i])
	#print "terms found after stemming: ", len(terms)

	fdist.update(list(set(terms)))
	#print fdist
	all_doc.append(terms)

fi11 = open('data/all_doc_with_uncommon', 'w')
pickle.dump(all_doc, fi11)
fi11.close()

fi13 = open('data/freq_dist', 'w')
pickle.dump(fdist, fi13)
fi13.close()

docs = []
count = 0
for doc in all_doc:
	count = count + 1
	#print len(doc)
	doc = [w for w in doc if fdist[w]>7000]
	#print len(doc)
	print "for document ", count, " after removing uncommon words, has ", len(doc), " terms."
	docs.append(doc)

all_terms = nltk.TextCollection(docs)
unique_terms = list(set(all_terms))
print "total number of terms found: ", len(all_terms)	
print "unique terms found: ", len(unique_terms)	

# all terms: 8939526
# unique terms: 385

# pprint.pprint(all_terms)

fi1 = open('data/all_doc_without_uncommon', 'w')
pickle.dump(docs, fi1)
fi1.close()

fi4 = open('data/unique_terms', 'w')
pickle.dump(unique_terms, fi4)
fi4.close()

fi5 = open('data/all_terms', 'w')
pickle.dump(all_terms, fi5)
fi5.close()


# collection = nltk.TextCollection(docs)
# print "Created a collection of", len(collection), "terms."

# terms = list(collection)
# terms = [w.lower() for w in terms]
# terms = [punctuation.sub("", w) for w in terms]
# terms = [w for w in terms if not w in stopwords.words('english')]
# terms = filter(len, terms)
# print "terms found after removing stop words: ", len(terms)

# for i in range(len(terms)):
# 	#print terms[i]
# 	terms[i] = PorterStemmer().stem_word(terms[i])
# 	#print terms[i]
# print "terms found after stemming: ", len(terms)

# unique_terms = list(set(terms))
# print "unique terms found: ", len(unique_terms)

#pprint.pprint(unique_terms)