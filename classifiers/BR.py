from numpy import *
import copy
from sklearn import linear_model

class BR() :
    '''
        Binary Relevance
        ----------------
    '''

    h = None
    L = -1

    def __init__(self, L=-1, h=linear_model.LogisticRegression()):
        '''
            Note: setting L does nothing here anymore! Everything is done in fit
            This L option can be deprecated!
        '''
        self.hop = h

    def fit(self, X, Y):
        N,L = Y.shape
        self.L = L
        self.h = [ copy.deepcopy(self.hop) for j in range(self.L)]

        #print "training ... [" ,
        for j in range(self.L):
            #print j , 
            self.h[j].fit(X, Y[:,j])
        #print "]"
        return self

    def partial_fit(self, X, Y):
        N,L = Y.shape

        for j in range(self.L):
            self.h[j].partial_fit(X, Y[:,j])

        return self

    def predict(self, X):
        '''
            return predictions for X
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for j in range(self.L):
            Y[:,j] = self.h[j].predict(X)
        return Y

    def predict_proba(self, X):
        '''
            return confidence predictions for X
            NOTE: for multi-label (binary) data only at the moment.
        '''
        N,D = X.shape
        P = zeros((N,self.L))
        for j in range(self.L):
            P[:,j] = self.h[j].predict_proba(X)[:,1]
        return P

def demo():
    import sys
    sys.path.append( '../core' )
    from tools import make_XOR_dataset

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    br = BR(L, linear_model.SGDClassifier(n_iter=100))
    br.fit(X, Y)
    # test it
    print(br.predict(X))
    print("vs")
    print(Y)

if __name__ == '__main__':
    demo()

