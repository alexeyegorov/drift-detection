from numpy import *
from time import clock

def exact(yt,yp):
    '''
        Error function
        --------------
    '''
    return (yp == yt) * 1

from evluation.metrics import J_index

def get_errors(Y,P,J=J_index):
    N,L = Y.shape
    E = zeros((N))
    for i in range(N):
        E[i] = J(Y[i,:].reshape(1,-1), P[i,:].reshape(1,-1)) 
    return E

def prequential_evaluation(X,Y,H,N_train):
    '''
        Prequential Evaluation
        ----------------------
        X                       instances
        Y                       labels
        H = [h_1,...,h_H]       a set of classifiers
        N_train                 number of instances for initial batch
        return the label predictions for each test instance, and the associated running time 
    '''
    M = len(H)
    T,L = Y.shape

    # split off an initial batch (maybe) ...
    Y_init = Y[0:N_train]
    X_init = X[0:N_train]

    # ... and then use the remainder, used for both incremental training and evaluation.
    Y = Y[N_train:]
    X = X[N_train:]

    E_pred = zeros((M,T-N_train,L))
    E_time = zeros((M,T-N_train))

    for m in range(M):
        #start_time = clock()
        H[m].fit(X_init,Y_init)
        #E_time[m,0] = clock() - start_time 

    for t in range(0,T-N_train):
        for m in range(M):
            start_time = clock()
            E_pred[m,t,:] = H[m].predict(X[t,:].reshape(1,-1))
            H[m].partial_fit(X[t,:].reshape(1,-1),Y[t,:].reshape(1,-1))
            E_time[m,t] += (clock() - start_time)

    return E_pred, E_time



