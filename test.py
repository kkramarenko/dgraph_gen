import numpy as np
#import graphviz
from time import time
#from datetime import timedelta
from digraph import trainset_gen
from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score
#from sklearn.tree import export_graphviz

#X = [[1, 0, 0, 0, 1],
#     [0, 0, 0, 1, 0],
#     [0, 1, 0, 1, 0]]

#Y1 = [[1, 3], [2], [1]]
N = 100
N_faulty = 10
trainset_length = 1000
n_estim = 1000

print "N: ", N
print "N_faulty: ", N_faulty
print "Length of trainset: ", trainset_length
print "n_estimators: ", n_estim

time_start = time()
Y, X = trainset_gen(trainset_length, N, N_faulty)
time_end = time()
print "Time of generating trainset: ", time_end - time_start, "seconds!" 

train_len = int(trainset_length * 0.7)
X_train = X[:train_len]
X_test = X[train_len+1:]
Y_train = Y[:train_len]
Y_test = Y[train_len+1:]

time_start = time()
model = RandomForestClassifier(n_estimators=n_estim)
model.fit(X_train, Y_train)
time_end =  time()
print "Time of learning: ", time_end - time_start, "seconds!"

ret = model.predict(X_test)
#print ret

#scores = model.score(X, Y)
#print scores

scores = label_ranking_average_precision_score(Y_test, ret)
print "Test set score: ", scores

ret = model.predict(X_train)
scores = label_ranking_average_precision_score(Y_train, ret)
print "Train set score: ", scores

