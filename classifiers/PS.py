from numpy import *
from sklearn import linear_model
from .LP import LP
from sklearn.linear_model import LogisticRegression, SGDClassifier
from .transforms import *

class PS(LP) :
    """
        Pruned Sets
        -----------
        Like LP, but prunes the collection of labelsets prior to training
    """

    p = 0

    def __init__(self, h=LogisticRegression(), p=0):
        LP.__init__(self, h=h)
        self.p = p

    def fit(self, X, Y):
        N, self.L = Y.shape
        y,selection,self.reverse = transform_BR2PS(Y,self.p)
        X = X[selection,:]
        #print y
        self.h.fit(X, y)
        return self


def demo():
    #from molearn.core.tools import make_XOR_dataset
    import sys
    sys.path.append( '../core' )
    from tools import make_XOR_dataset

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    ps = PS()
    ps.fit(X, Y)
    # test it
    print(ps.predict(X))
    print("vs")
    print(Y)

if __name__ == '__main__':
    demo()

