import numpy as np
#import graphviz
from time import time
#from datetime import timedelta
from digraph import trainset_gen
from keras.models import Sequential
#from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import label_ranking_average_precision_score
#from sklearn.tree import export_graphviz

def my_metric(y_true, y_pred):
    return label_ranking_average_precision_score(y_true,y_pred);

#X = [[1, 0, 0, 0, 1],
#     [0, 0, 0, 1, 0],
#     [0, 1, 0, 1, 0]]

#Y1 = [[1, 3], [2], [1]]
N = 50
N_faulty = 20
trainset_length = 1000
#n_estim = 100

print "N: ", N
print "N_faulty: ", N_faulty
print "Length of trainset: ", trainset_length
#print "n_estimators: ", n_estim
#print "max_depth: ", mx_dp

#np.random.seed(7)

time_start = time()
Y, X = trainset_gen(trainset_length, N, N_faulty)
time_end = time()
print "Time of generating trainset: ", time_end - time_start, "seconds!" 

train_len = int(trainset_length * 0.7)
X_train = X[:train_len]
X_test = X[train_len+1:]
Y_train = Y[:train_len]
Y_test = Y[train_len+1:]

X_train1 = X_train.reshape(-1, N, N, 1)
X_test1 = X_test.reshape(-1, N, N, 1)
print X_train1.shape
print X_test1.shape

time_start = time()
#max_features=N*10
model = Sequential()
model.add(Conv2D(8, (3, 3), activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(12, (7, 7), activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(N, activation='linear'))
model.add(LeakyReLU(alpha=0.1))

#model.add(Dense(N*3, input_dim=N*N, activation='relu'))
#model.add(Dense(N*3, activation='relu'))
#model.add(Dense(N, activation='relu'))
model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])

model.fit(X_train1, Y_train, epochs=15, batch_size=100)
time_end =  time()
print "Time of learning: ", time_end - time_start, "seconds!"

ret = model.predict(X_test1)
#print ret

#scores = model.score(X, Y)
#print scores

scores = label_ranking_average_precision_score(Y_test, ret)
print "Test set score: ", scores

ret = model.predict(X_train1)
scores = label_ranking_average_precision_score(Y_train, ret)
print "Train set score: ", scores

