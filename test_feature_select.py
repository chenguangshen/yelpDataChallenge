from sklearn.feature_selection import SelectKBest, f_classif

X = [[0.2, 0.3], [1.2, 3.5]]
Y = [1, 2]
clf = SelectKBest( k=1)
clf = clf.fit(X, Y)

print clf.scores_
