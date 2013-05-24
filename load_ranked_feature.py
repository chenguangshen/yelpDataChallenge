import pickle
import pprint

fi111 = open('data/var_importance_lasso', 'r')
var = pickle.load(fi111)
fi111.close()

print len(var)
# pprint.pprint(var)

top_word = sorted(range(len(var)), key=lambda x:var[x])
print len(top_word)

pprint.pprint(top_word[:20])
