from sklearn.ensemble import RandomForestRegressor
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestRegressor(n_estimators=10, compute_importances=True)
clf = clf.fit(X, Y)

print clf.feature_importances_
