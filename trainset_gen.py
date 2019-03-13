import numpy as np
from digraph import trainset_gen
from tempfile import NamedTemporaryFile


length = 100
N = 50
N_faulty = 1
trainset_count = 10



for i in range(1, trainset_count+1):
    filename_X = "trainset_" + str(length) + "_" + str(N) + "_" + str(N_faulty) + "_" + str(i) + "_X" 
    filename_Y = "trainset_" + str(length) + "_" + str(N) + "_" + str(N_faulty) + "_" + str(i) + "_Y"
    
    Y, X = trainset_gen(length, N, N_faulty)
    
    filex = open(filename_X, "wb")
    np.save(filex, X)
    filex.close()

    filey = open(filename_Y, "wb")
    np.save(filey, Y)
    filey.close()

#filex = open(filename_X, "rb")
#X1 = np.load(filex)
#filex.close()
#print X
#print X1
#print type(X1)
#print X1.shape

