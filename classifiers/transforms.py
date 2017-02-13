from numpy import *

"""
    Transforms
    ---------
    Functions for transforming back and forth between binary (multi-label) and multi-class (single-label) representations,
    all of which contstitute 'meta-label' aproaches.
    TODO: Later for bigger data, we may need to work with sparse representations instead of dense binary representations.
"""

def prune_combinations(count,p=0,p_max=2):
    ''' 
    Prune Combinations
    -----------------------------------------
    count: {
        "01":50,
        "10":203,
        "11":2
    }
    p: the minimum support count (i.e., pruned sets) 
    p_max: prune to have at most p_max classes
    (only one of p,p_max may be above 0 at a time)
    '''
    import operator
    sorted_count = sorted(count.items(), key=operator.itemgetter(1))
    n = len(sorted_count)
    items_to_cut = sorted_count[0:min(n-p_max,n)]
    
    if p > 0:
        # Standard PS pruning
        for key,c in items_to_cut:
            if c <= p:
                count.pop(key)
            else: 
                break

    if p_max > 0:
        # Pruning down to binary
        for key,c in items_to_cut:
            count.pop(key)
    return count


def count_combinations(Y):
    ''' make a dictionary count of occurences in Y '''
    counts = {}
    N,L = Y.shape
    for i in range(N):
        k = str(Y[i])
        if k not in counts:
            counts[k] = 1
        else:
            counts[k] = counts[k] + 1
    return counts

def transform_BR2PS(Y,p=0,p_max=0,n=0):
    ''' 
    PS transform: BR matrix to PS vector (with mapping key) 
    --------------------------------------------------------
    Y: the label matrix
    p: the pruning to do (i.e., pruned sets)

    1. count 
    2. prune 
    3. map / copy

    returns
    --------
        y: e.g.,  [[1,0,0],...,[1,1,0]]
        selection: e.g., [1,2,3,5,6,8]    (so that the X space can be mapped properly)
        mapping: e.g., 4 -> [1,0,1]
    '''

    # 1. count
    counts = count_combinations(Y)
    # 2. prune
    counts = prune_combinations(counts,p,p_max)
    # 3. map

    N,L = Y.shape
    Y_LP = zeros(N,dtype=int)
    c = -1
    mapping = {}
    reverse = {}
    N_new = 0
    selection = []
    for i in range(N):
        k = str(Y[i])
        if k in counts:
            # add it 
            if k not in mapping:
                # (first, make the key if it does not exist)
                c = c + 1
                mapping[k] = c
                reverse[c] = array(Y[i,:])
            Y_LP[N_new] = mapping[k]
            N_new = N_new + 1
            selection.append(i)
        elif n > 0:
            ## copy it
            print("COPY (@TODO)")

    return Y_LP[0:N_new],selection,reverse

def transform_BR2MC(Y):
    ''' 
    LP TRANSFORM: BR matrix to LP vector (with mapping key) 
    -----------------------------------------
    Y: the binary label matrix

    i.e., standard LP transformation

    returns the LP-transformation and a mapping to reverse it (and recover the original binary labels)

    '''
    N,L = Y.shape
    Y_LP = zeros(N,dtype=int)
    c = -1
    mapping = {}
    reverse = {}
    for i in range(N):
        k = str(Y[i])
        if k not in mapping:
            c = c + 1
            mapping[k] = c
            reverse[c] = array(Y[i,:])
        Y_LP[i] = mapping[k]

    return Y_LP,reverse
    
def transform_MC2BR(y,L,reverse):
    ''' 
    LP reverse-TRANSFORM: LP vector (with mapping key) to BR matrix 
    ---------------------------------------------------------------
    (can also use with PS)
    '''

    N = len(y)
    Y = zeros((N,L),dtype=int)
    for i in range(N):
        Y[i,:] = reverse[y[i]]
    return Y

def transform_BR2ML(Y,k):
    ''' 
    RAkELd-ML/Rd/meta-label-TRANSFORM: BR matrix to k-LP vectors (with mapping keys) 
    ---------------------------------------------------------------------
    k: the size of subsets, for example L = 6, k = 4, the there will be in total n = ceil(6/4) = 2


    Y = [0,1,1;
         0,1,0]

    Y_ML = [0,0;
            0,0]
        

    '''
    N,L = Y.shape
    n = int(ceil(L*1./k))
    Y_ML = zeros((N,n),dtype=int)
    reverse_mapping = [{} for n_ in range(n)]
    for n_ in range(0,n):
        i_start = n_*k
        i_end = min(i_start+k,L)
        Y_ML[:,n_],reverse_mapping[n_] = transform_BR2MC(Y[:,i_start:i_end])
    return Y_ML,reverse_mapping

def transform_ML2BR(Y_ML,reverse_mapping,L,k):
    ''' RAkELd-ML/Rd/meta-label reverse-TRANSFORM '''
    N,n = Y_ML.shape
    Y = zeros((N,L),dtype=int)
    for n_ in range(n):
        i_start = n_*k
        i_end = min(i_start+k,L)
        L_ = i_end - i_start
        Y[:,i_start:i_end] = transform_MC2BR(Y_ML[:,n_],L_,reverse_mapping[n_])
        #for j in len(y_k):
        #    for i in range(N):
        #        Y[i,j] = Y[i,j] + y_k[j]
    return Y

def transform_BR2bML(Y,k):
    ''' 
    Binary Meta-label TRANSFORM: 
    ---------------------------------------------------------------------
    This is basically RAkELd, but each sub-problem is PS, pruned down to only two!
        001
        101
        001
        100

        Y_ML in {001,!001}

    same as transform_BR2ML, but for binary meta labels 
    actually, because of the pruning, two lines are different -- we return a LIST of arrays, not a 2d arary
    '''
    N,L = Y.shape
    n = int(ceil(L*1./k))
    Y_ML = [None for n_ in range(n)] #zeros((N,n),dtype=int)                                # <-- this line also
    reverse_mapping = [{} for n_ in range(n)]
    #for k_ in range(0,L,k):
    for n_ in range(0,n):
        i_start = n_*k
        i_end = min(i_start+k,L)
        Y_ML[n_],selection,reverse_mapping[n_] = transform_BR2PS(Y[:,i_start:i_end],p=0,p_max=2,n=0)    # <-- only line that is different
    return Y_ML,reverse_mapping

def transform_bML2BR(y, mapping, L):
    ''' Transform a binary meta label back into bits, this is lossy!!! '''
    N = len(y)
    Y = zeros((N,L),dtype=int)
    for i in range(N):
        k = y[i]
        Y[i] = mapping[k]
    return Y


def transform_BR2PW(Y):
    ''' make a pair-wise meta-label for each pair '''
    N, L = Y.shape
    L_ = L*(L-1)/2
    YY = zeros((N,L_),dtype=int)
    m = {}
    for i in range(N):
        c = 0
        for j in range(L):
            for k in range(j+1,L):
                b = (Y[i,j]*2) + Y[i,k]
                if b == 2 or b == 0:
                    import random
                    YY[i,c] = random.randrange(2)
                if Y[i,k] == 1:
                    YY[i,c] = 0
                elif Y[i,k] == 1:
                    YY[i,c] = 1
                m[c] = (j,k)
                c = c + 1
    return YY,m

def transform_BR2FW(Y):
    ''' make a four-wise meta-label for each pair '''
    N, L = Y.shape
    YY = zeros((N,L*(L-1)/2),dtype=int)
    m = {}
    for i in range(N):
        c = 0
        for j in range(L):
            for k in range(j+1,L):
                YY[i,c] = (Y[i,j]*2) + Y[i,k]
                m[c] = (j,k)
                c = c + 1
    return YY,m

def transform_FW2BR(YY, m, L):
    ''' return from a four-wise meta-label for each pair '''
    N, L_ = YY.shape
    Y = zeros((N,L))
    for i in range(N):
        for key in range(L_):
            j,k = m[key]
            if YY[i,key] == 1:
                Y[i,j] = Y[i,j] + 0 
                Y[i,k] = Y[i,k] + 1
            if YY[i,key] == 2:
                Y[i,j] = Y[i,j] + 1 
                Y[i,k] = Y[i,k] + 0
            if YY[i,key] == 3:
                Y[i,j] = Y[i,j] + 1 
                Y[i,k] = Y[i,k] + 1
    return Y

def tests():
    #               0                                     1                          -1        2
    A = array([[1,0,1,0,1], [1,0,1,0,1], [1,0,1,0,1], [0,0,0,1,1], [0,0,0,1,1], [0,0,0,0,1], [1,0,0,1,1], [1,0,0,1,1]])
    y,selection,r_map = transform_BR2PS(A,1)
    print(y)
    print(selection)
    print(r_map)
    print(y)
    print(selection)
    print(r_map)
    print("Y")
    print(A)
    y,reverse = transform_BR2bML(A,2)
    print(y)
    print(reverse)
    print(y, reverse)
    print("y")
    print(y)
    A_ = transform_bML2BR(y,reverse,5)
    print("Y'")
    print(A_)

if __name__ == '__main__':
    tests()
