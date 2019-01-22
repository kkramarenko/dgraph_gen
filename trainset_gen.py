import numpy as np
from digraph import trainset_gen
from tempfile import NamedTemporaryFile
#for i in range(1,11):
#    print i
length = 1
N = 10
N_faulty = 1
Y, X = trainset_gen(length, N, N_faulty)

filename_X = "trainset_" + str(length+1) + "_" + str(N) + "_" + str(N_faulty) + "_X" 
filename_Y = "trainset_" + str(length+1) + "_" + str(N) + "_" + str(N_faulty) + "_Y"

filex = open(filename_X, "wb")
np.save(filex, X)
filex.close()

filex = open(filename_X, "rb")
X1 = np.load(filex)
filex.close()
print X
print X1
print type(X1)
print X1.shape

