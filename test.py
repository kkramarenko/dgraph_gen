import numpy as np
from digraph import graph_gen
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score

X = [[1, 0, 0, 0, 1],
     [0, 0, 0, 1, 0],
     [0, 1, 0, 1, 0]]

Y = [[1, 3], [2], [1]]

print X
print Y

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(Y)

print Y

model = RandomForestClassifier()
model.fit(X, Y)

ret = model.predict(X)
print ret

scores = model.score(X, Y)
print scores

scores = label_ranking_average_precision_score(Y, ret)
print scores
#graph_gen(51)

