import nltk
import sys
import pprint
import pickle
import numpy

data_path = 'E:\workspace\cs246_data\\'

fin1 = open(data_path + 'all_doc_without_uncommon', 'r')
docs = pickle.load(fin1)
fin1.close()
print len(docs)

fin2 = open(data_path + 'unique_terms', 'r')
unique_terms = pickle.load(fin2)
fin2.close()
print len(unique_terms)

# fin3 = open(data_path + 'all_doc_without_uncommon_without_empty', 'w')
# pickle.dump(docs, fin3)
# fin3.close()

collection = nltk.TextCollection(docs)

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

fin3 = open(data_path + 'tf_idf_all', 'w')
pickle.dump(vectors, fin3)
fin3.close()