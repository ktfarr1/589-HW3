import numpy as np
    
def computeXY(rdd):    
    '''
    Compute the features times targets term
    needed by OLS. Return as a numpy array
    of size (41,)
    '''
    return rdd.map(lambda x: np.hstack((x[:-1]*x[-1],x[-1]))).reduce(lambda x,y: x+y)

def computeXX(rdd):
    '''
    Compute the outer product term
    needed by OLS. Return as a numpy array
    of size (41,41)
    '''    
    return rdd.map(lambda x: np.hstack((x[:-1],[1]))).map(lambda x: np.outer(x,x)).reduce(lambda x,y: x+y)

    
def computeWeights(rdd):  
    '''
    Compute the linear regression weights.
    Return as a numpy array of shape (41,)
    '''
    xx = computeXX(rdd)
    xy = computeXY(rdd)
    xxinv = np.linalg.inv(xx)
    weight = np.dot(xxinv, xy)
    return weight
    
def computePredictions(w,rdd):  
    '''
    Compute predictions given by the input weight vector
    w. Return an RDD with one prediction per row.
    '''
    return rdd.map(lambda x: np.hstack((x[:-1],[1]))).map(lambda x: np.dot(w,x))
    
def computeError(w,rdd):
    '''
    Compute the MAE of the predictions.
    Return as a float.
    '''
    c = float(rdd.map(lambda x: 1).reduce(lambda x,y : x+y))
    return rdd.map(lambda x: np.hstack( (x[-1], np.dot(w,np.hstack((x[:-1],[1])))  ))).map(lambda x: np.absolute(x[0]-x[1])*(1/c)).reduce(lambda x,y: x+y)
