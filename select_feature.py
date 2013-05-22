import nltk
import sys
import pprint
import pickle
import numpy

data_path = 'data/'
#data_path = '/Users/cgshen/Dataset/cs246_data/'

fin1 = open(data_path + 'tf_idf_top_20_rf', 'r')
tf_idf_top_20 = pickle.load(fin1)
fin1.close()
print len(tf_idf_top_20)

fin2 = open(data_path + 'nontext_features/rating_feature', 'r')
review_rate = pickle.load(fin2)
fin2.close()
print len(review_rate)

fin3 = open(data_path + 'nontext_features/length_feature', 'r')
review_length = pickle.load(fin3)
fin3.close()
print len(review_length)

fin4 = open(data_path + 'nontext_features/date_feature', 'r')
review_date = pickle.load(fin4)
fin4.close()
print len(review_date)

fin5 = open(data_path + 'nontext_features/cool_feature', 'r')
review_cool_count = pickle.load(fin5)
fin5.close()
print len(review_cool_count)

fin6 = open(data_path + 'nontext_features/funny_feature', 'r')
review_funny_count = pickle.load(fin6)
fin6.close()
print len(review_funny_count)

fin11 = open(data_path + 'user_features/avg_rating_feature', 'r')
user_avg_rate = pickle.load(fin11)
fin11.close()
print len(user_avg_rate)

fin12 = open(data_path + 'user_features/total_review_feature', 'r')
user_total_review = pickle.load(fin12)
fin12.close()
print len(user_total_review)

fin13 = open(data_path + 'user_features/total_useful_feature', 'r')
user_total_useful = pickle.load(fin13)
fin13.close()
print len(user_total_useful)

fin14 = open(data_path + 'user_features/total_cool_feature', 'r')
user_total_cool = pickle.load(fin14)
fin14.close()
print len(user_total_cool)

fin15 = open(data_path + 'user_features/total_funny_feature', 'r')
user_total_funny = pickle.load(fin15)
fin15.close()
print len(user_total_funny)

features = []
for i in range(229907):
	if i % 100 == 0:
		print i
	t = []
	t.extend(tf_idf_top_20[i])
	t.append(review_rate[i])
	t.append(review_length[i])
	t.append(review_date[i])
	t.append(review_cool_count[i])
	t.append(review_funny_count[i])
	t.append(user_avg_rate[i])
	t.append(user_total_review[i])
	t.append(user_total_useful[i])
	t.append(user_total_cool[i])
	t.append(user_total_funny[i])
	#print t
	features.append(t)

fout = open(data_path + 'all_30_features_ranked_by_rf', 'w')
pickle.dump(features, fout)
fout.close()

############################################################

#pprint.pprint(features)

# total = {}
# count = 0
# for e in tf_idf:
# 	if count % 100 == 0:
# 		print count
# 	count = count + 1
# 	for i in range(len(e)):
# 		if i in total:
# 			total[i] = total[i] + e[i]
# 		else:
# 			total[i] = e[i]

# print len(total)
# #pprint.pprint(total)

# sorted_index = sorted(total, key=total.get)
# sorted_index = sorted_index[:20]
# sorted_index = [209,
#  102,
#  42,
#  356,
#  263,
#  194,
#  75,
#  81,
#  375,
#  364,
#  353,
#  314,
#  96,
#  216,
#  26,
#  56,
#  289,
#  174,
#  197,
#  378]
# pprint.pprint(sorted_index)

#################################################################