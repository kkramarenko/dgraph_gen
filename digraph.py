import numpy as np

def syndrome_gen(N, N_faulty):
    """Function for generating syndrome of system.

    Args:
        N: Total number of nodes of computer system.
        N_faulty: Number of faulty nodes.

    Returns:
        Set of fafulty_nodes and syndrome corresponded to system state.    
    """
    
#    print N
    syndrome = np.empty((N, N), dtype='int8')

    syndrome[:] = -1

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
    
    tmp = np.array(list(faulty_set)[:N_faulty]) - 1
    faulty_set = set(tmp)
#    print faulty_set
#    print faulty_nodes
#    print len(faulty_nodes)

#   Generate syndrome     
    for idx1 in range(N):
        for idx2 in range(N):
            if idx1 != idx2:
                if idx1 in faulty_set:
                    syndrome[idx1][idx2] = np.random.randint(low=0, high=2)
                else:    
                    if idx2 in faulty_set:
                        syndrome[idx1][idx2] = 1
                    else:
                        syndrome[idx1][idx2] = 0

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
    X = np.zeros(N*N, dtype='int8')
#    print Y.shape
#    print X.shape
    while length > 0:
        tmp_Y, tmp_X = syndrome_gen(N, N_faulty)
        Y = np.vstack([Y, tmp_Y])
        X = np.vstack([X, tmp_X])
        length -= 1

    return Y, X

#trainset_gen(1, 5, 2)

