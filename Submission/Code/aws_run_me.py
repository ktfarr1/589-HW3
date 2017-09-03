from pyspark import SparkContext, SparkConf
import numpy as np
import time

#============
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
   

#============

#Convert rdd rows to numpy arrays
def numpyify(rdd):
    return rdd.map(lambda x: np.array(map(lambda y: float(y),x.split(","))))

#sc     = spark.sparkContext
master = "yarn"
times=[]


#Flush yarn defaul context
sc = SparkContext(master, "aws_run_me")
sc.stop()

for i in [1,2,3,4,5,6,7,8]:

    conf = SparkConf().set("spark.executor.instances",i).set("spark.executor.cores",1).set("spark.executor.memory","2G").set("spark.dynamicAllocation.enabled","false")
    sc = SparkContext(master, "aws_run_me", conf=conf)
    sc.setLogLevel("ERROR")

    start=time.time()

    rdd_test = numpyify(sc.textFile("s3://589hw03/test_data_ec2.csv"))        
    rdd_train = numpyify(sc.textFile("s3://589hw03/train_data_ec2.csv"))

    w = computeWeights(rdd_train)
    err = computeError(w,rdd_test)
    
    this_time =  time.time()-start
    print "\n\n\nCores %d: MAE: %.4f Time: %.2f"%(i, err,this_time)
    times.append([i,this_time])

    sc.stop()

print "\n\n\n\n\n"
print times
