import numpy as np

def count(rdd):
    '''
    Computes the number of rows in rdd.
    Returns the answer as a float.
    '''

    return rdd.map(lambda x: 1).reduce(lambda x,y : x+y)

def mean(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (D,)
    Returns the sample mean of each column of rdd as a numpy array of shape (D,)
    '''
    n = float(count(rdd))
    return rdd.reduce(lambda x,y: x+y)/n

def std(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (D,)
    Returns the sample standard deviation of 
    each column of rdd as a numpy array of shape (D,)
    '''
    n = count(rdd)
    m = mean(rdd)
    return np.sqrt(rdd.map(lambda x: (np.square(x-m)/n)).reduce(lambda x,y : x+y))

def dot(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (2,)
    Returns the inner (dot) product between the columns as a float.
    '''  
    return rdd.map(lambda x: x[0]*x[1]).reduce(lambda x,y: x+y)

def corr(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (2,)
    Returns the sample Pearson's correlation between the columns as a float.
    '''  
    n = count(rdd)
    # m = mean(rdd)[0:2]
    # s = std(rdd)[0:2]
    # return (dot(rdd)-(n*m[0]*m[1]))/(s[0]*s[1]*(n-1))
    d = dot(rdd)
    sumx = rdd.map(lambda x: x[0]).reduce(lambda x,y: x+y)
    sumx2 = rdd.map(lambda x: np.square(x[0])).reduce(lambda x,y: x+y)
    sumy = rdd.map(lambda x: x[1]).reduce(lambda x,y: x+y)
    sumy2 = rdd.map(lambda x: np.square(x[1])).reduce(lambda x,y: x+y)
    
    return (d - (sumx*sumy)/n)/np.sqrt((sumx2 - (np.square(sumx))/n) * (sumy2 - (np.square(sumy))/n))
    # (sumy2 - (np.square(sumy))/n))

    
def distribution(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (1,)
    and that the values in rdd are whole numbers in [0,K] for some K.
    Returns the empirical distribution of the values in rdd
    as an array of shape (K+1,)
    '''
    n = float(count(rdd))
    return rdd.map(lambda x: (x, 1)).reduceByKey(lambda x,y: x+y).map(lambda x: x[1]/n).collect()
