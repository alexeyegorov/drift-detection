from numpy import *

def Hamming_loss(Ytest,Ypred):
    ''' Hamming loss aka Hamming distance '''
    return 1.-Hamming_score(Ytest,Ypred)

def Hamming_score(Ytest,Ypred):
    ''' Hamming score aka Hamming match '''
    N_test,L = Ytest.shape
    return sum((Ytest == Ypred) * 1.) / N_test / L

def Hamming_matches(Ytest,Ypred):
    N_test,L = Ytest.shape
    return sum((Ytest == Ypred) * 1.,axis=0) / N_test 

def Hamming_losses(Ytest,Ypred):
    return 1.-Hamming_matches(Ytest,Ypred)

from sklearn.metrics import log_loss


def Log_loss(Ytest,Ydist):
    return log_loss(Ytest, Ydist, eps=1e-15, normalize=True)
#    N_test,L = Ytest.shape
#    return sum((Ytest == Ypred) * 1.) / N_test / L

def J_index(Ytest,Ypred):
    N_test,L = Ytest.shape
    s = 0.0
    for i in range(N_test):
        inter = sum((Ytest[i,:] * Ypred[i,:]) > 0) * 1.
        union = sum((Ytest[i,:] + Ypred[i,:]) > 0) * 1.
        if union > 0:
            s = s + ( inter / union )
        elif sum(Ytest[i,:]) == 0:
            s = s + 1.
    return s * 1. / N_test

def Exact_match(Ytest,Ypred):
    N_test,L = Ytest.shape
    return sum(sum((Ytest == Ypred) * 1,axis=1)==L) * 1. / N_test

def printEvalHeader():
    print("Algorithm            Jacc. Hamm. Exact Time  ")

def printEval(Ytest,Ypred,name="Method",time = 0.0):
    print("%-20s %.3f %.3f %.3f %0.1f" % (name, J_index(Ytest,Ypred), Hamming_loss(Ytest,Ypred), Exact_match(Ytest,Ypred), time))

def Edit_distance(Ytest,Ypred):
    ''' Average edit distance '''
    N_test,L = Ytest.shape
    s = 0.
    for i in range(N_test):
        s = s + edit_distance(Ytest[i,:],Ypred[i,:])
    return s * 1. / N_test

def h_loss(ytest,ypred):
    ''' note: required by edit_distance to only return bits (not average bits / L) '''
    return sum(ytest != ypred)

def Hamming_distances(Ytest,Ypred):
    ''' probably only to be used for sequential data '''
    N_test,L = Ytest.shape
    return sum((Ytest != Ypred) * 1.,axis=0) / N_test

def Edit_distances(Ytest,Ypred):
    N_test,L = Ytest.shape
    d = zeros(L)
    for j in range(L):
        d[j] = Edit_distance(Ytest[:,0:j+1],Ypred[:,0:j+1])
    return d / arange(1,L+1)

def edit_distance(y, p):
    ''' 
        aka Levenshtein
        From Wikipedia article; Iterative with two matrix rows. 
    '''
    if h_loss(y,p) == 0: return 0
    elif len(y) == 0: return len(p)
    elif len(p) == 0: return len(y)
    v0 = [None] * (len(p) + 1)
    v1 = [None] * (len(p) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(y)):
        v1[0] = i + 1
        for j in range(len(p)):
            cost = 0 if y[i] == p[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
            
    return v1[len(p)]

#y = array([0,8,2,9,7])
#p = array([8,2,9,7,0])
#print h_loss(y,p)
#print edit_distance(y,p)
