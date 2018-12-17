import numpy as np
import graphviz
#from digraph import graph_gen
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.tree import export_graphviz

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

tmp_X = np.array([1,1,1])
print tmp_X.reshape(1,-1)
ret = model.predict(tmp_X)
print ret

#scores = model.score(X, Y)
#print scores

#scores = label_ranking_average_precision_score(Y, ret)
#print scores

#dot_data = export_graphviz(model.estimators_[0], out_file=None)

#graph = graphviz.Source(dot_data)
#graph.render('TEST', view=True)
#graph_gen(51)

