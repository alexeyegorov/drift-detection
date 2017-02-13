from numpy import *
from .transforms import transform_BR2MC, transform_MC2BR
from sklearn.linear_model import LogisticRegression, SGDClassifier
import copy

class LP() :
    '''
        Label Powerset Method
        ---------------------
        Note: Multi-label only (binary outputs).
    '''

    h = None
    L = -1
    reverse = {}

    def __init__(self, L=-1, h=LogisticRegression()):
        # TODO: L is deprecated here.
        self.h = copy.deepcopy(h)

    def fit(self, X, Y):
        N, self.L = Y.shape
        y,self.reverse = transform_BR2MC(Y)
        #print "train with " , len(self.reverse) , " unique classes"
        self.h.fit(X, y)
        return self

    def predict(self, X):
        '''
            return predictions for X
        '''
        y = self.h.predict(X)
        N,D = X.shape
        Y = transform_MC2BR(y,self.L,self.reverse)
        return Y

    def predict_proba(self, X):
        '''
            return predictions for X, with a probability associated, 'LPt'-style
        '''
        P = self.h.predict_proba(X)
        N,C = P.shape
        Y = self.predict(X)
        Y_proba = zeros(Y.shape)
        for i in range(N):
            for c in range(C):
                Y_proba[i] = Y_proba[i] + Y[i] * P[c] 
        Y_proba = Y_proba * 1./C
        return Y_proba

def demo():
    #from molearn.core.tools import make_XOR_dataset
    import sys
    sys.path.append( '../core' )
    from tools import make_XOR_dataset

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    lp = LP()
    lp.fit(X, Y)
    # test it
    print(lp.predict_proba(X))
    print("vs")
    print(Y)

if __name__ == '__main__':
    demo()

