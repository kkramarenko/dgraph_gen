import numpy as np
#import graphviz
from time import time
#from datetime import timedelta
from digraph import trainset_gen
from keras.models import Sequential
#from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Dense
from sklearn.metrics import label_ranking_average_precision_score
#from sklearn.tree import export_graphviz

#X = [[1, 0, 0, 0, 1],
#     [0, 0, 0, 1, 0],
#     [0, 1, 0, 1, 0]]

#Y1 = [[1, 3], [2], [1]]
N = 50
N_faulty = 1
trainset_length = 100
epochs_num = 15
#n_estim = 100

print "N: ", N
print "N_faulty: ", N_faulty
print "Length of trainset: ", trainset_length
#print "n_estimators: ", n_estim
#print "max_depth: ", mx_dp

np.random.seed(7)

time_start = time()
#Y, X = trainset_gen(trainset_length, N, N_faulty)
filex = open("trainset_100_50_1_1_X", "rb")
X = np.load(filex)
filex.close()

filey = open("trainset_100_50_1_1_Y", "rb")
Y = np.load(filey)
filey.close()
time_end = time()

print X.shape
print Y.shape

train_len = int(trainset_length * 0.7)
X_train = X[:train_len]
X_test = X[train_len+1:]
Y_train = Y[:train_len]
Y_test = Y[train_len+1:]

#print type(X_train[0][0]) 

time_start = time()
#max_features=N*10
model = Sequential()
model.add(Dense(N*3, input_dim=N*N, activation='relu'))
model.add(Dense(N*3, activation='relu'))
model.add(Dense(N, activation='relu'))
model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=epochs_num, batch_size=100)
time_end =  time()
print "Time of learning: ", time_end - time_start, "seconds!"

ret = model.predict(X_test)
#print ret

#scores = model.score(X, Y)
#print scores

test_scores = label_ranking_average_precision_score(Y_test, ret)
print "Test set score: ", test_scores

ret = model.predict(X_train)
train_scores = label_ranking_average_precision_score(Y_train, ret)
print "Train set score: ", train_scores

