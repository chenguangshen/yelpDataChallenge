import pickle
import pprint

fi111 = open('data/var_importance_rf', 'r')
var = pickle.load(fi111)
fi111.close()

print len(var)
pprint.pprint(var)