import numpy as np
from time import time
from multiprocessing.pool import ThreadPool

def syndrome_gen(N, N_faulty):
    """Function for generating syndrome of system.

    Args:
        N: Total number of nodes of computer system.
        N_faulty: Number of faulty nodes.

    Returns:
        Set of fafulty_nodes and syndrome corresponded to system state.    
    """
    
#    print N
#    syndrome = np.empty((N, N), dtype='int8')
#    syndrome[:] = -1
    vec = np.empty(N)
    vec.fill(-1)
    syndrome = np.diag(vec)

#    tmp = np.reshape(syndrome, N*N)
#    print syndrome
#    print tmp

    N_faulty = np.random.randint(low=1, high=N_faulty+1)
#   Selecting randomly nodes which ones will be faulty
    faulty_set = set()
    while len(faulty_set) < N_faulty:
        tmp = np.random.random_integers(low=1, high=N, size=N_faulty)
        faulty_set.update(tmp)
#        print faulty_set
#        print len(faulty_set)
#        print tmp
    
#    print faulty_set
    tmp = np.array(list(faulty_set)[:N_faulty]) - 1
    faulty_set = set(tmp)
#    print faulty_set
#    print faulty_nodes
#    print len(faulty_nodes)

#   Generate syndrome
    syndrome[:, tmp] = 1;
    for idx1 in faulty_set:
        for idx2 in range(N):
            syndrome[idx1][idx2] = np.random.randint(low=0, high=2)
        syndrome[idx1][idx1] = -1
#    for idx1 in range(N):
#        for idx2 in range(N):
#            if idx1 != idx2:
#                if idx1 in faulty_set:
#                    syndrome[idx1][idx2] = np.random.randint(low=0, high=2)
#                else:    
#                    if idx2 in faulty_set:
#                        syndrome[idx1][idx2] = 1
#                    else:
#                        syndrome[idx1][idx2] = 0
    
#    print "Time of syndrome: ", time() - time_start
#    print syndrome

    faulty_nodes = np.zeros(N, dtype='int8')
    for idx in range(N):
        if idx in faulty_set:
            faulty_nodes[idx] = 1
    
    syndrome = syndrome.reshape(-1, )
    return faulty_nodes, syndrome

def trainset_gen(length, N, N_faulty):
    """Function for generating trainset.

    Args:
        Length: Total length of training set
        N: Total number of nodes of computer system.
        N_faulty: Number of faulty nodes.

    Returns:
        Training inputs and outputs.    
    """

    Y = np.zeros(N, dtype='int8')
    X = np.empty(N, dtype='int8')
    X.fill(-1)
    X = np.diag(X)
    X = X.reshape(-1,)
#    print Y.shape
#    print X.shape
    while length > 0:
        tmp_Y, tmp_X = syndrome_gen(N, N_faulty)
        Y = np.vstack([Y, tmp_Y])
        X = np.vstack([X, tmp_X])
        length -= 1

    return Y, X

def parallel_gen(thread_num, length, N, N_faulty):
    pool = ThreadPool(thread_num)
    results = []
   
    time_start = time()
    for thread_idx in range(0, thread_num):
        results.append(pool.apply_async(trainset_gen,(int(length/thread_num), N, N_faulty)))

    for res  in results:
        tmp = res.get()
        tmp_X = np.array(tmp[0])
        tmp_Y = np.array(tmp[1])

        print tmp_X.shape, tmp_Y.shape
            
    time_end = time()
#    print "Gen time: ", time_end - time_start 
    pool.close()
    pool.join()

#parallel_gen(8, 30000, 1000, 3)
#time_start = time()
#trainset_gen(10000, 100, 3)
#parallel_gen(4, 10000, 1000, 3)
#print "Time: ", time() - time_start
#file = open("rez.txt", "w")
#file.write("Time:")
#file.write(time() - time_start)
#file.close()


