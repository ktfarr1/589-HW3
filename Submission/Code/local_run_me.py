from mySpark import RDD
from mySpark import SparkContext
import numpy as np
import warmup
import ols


SC  = SparkContext()

#Create data for Question 1
np.random.seed(0)
X    = np.hstack((np.random.randint(0,5,(100,1)),np.random.rand(100,1),np.random.randn(100,1)))
rdd1 = SC.parallelize(X) 

print "Question 1:"

print "1.1 Count:", warmup.count(rdd1)

print "1.2 Mean:", warmup.mean(rdd1)

print "1.3 Std:", warmup.std(rdd1)

print "1.4 Dot:", warmup.dot(rdd1)

print "1.5 Corr:", warmup.corr(rdd1)

rdd2 = rdd1.map(lambda x: x[0])
print "1.6 Distribution:", warmup.distribution(rdd2)


#Load data for question 2 assuming you are running local_run_me.py 
#from the Submission/Code directory. Do not change path to data.

path = '../../Data/movielens/'
train = np.loadtxt(path + 'train_data.csv', delimiter = ',')
train[:,-2] = np.random.randn(train.shape[0])/100.0
test = np.loadtxt(path + 'test_data.csv', delimiter = ',')
rdd_tr = SC.parallelize(train)
rdd_te = SC.parallelize(test)

print "Question 2:"

print "2.1 computeXY:", ols.computeXY(rdd_tr)  

print "2.2 computeXX:", ols.computeXX(rdd_tr)

w=ols.computeWeights(rdd_tr)
print "2.3 computeWeights:", w 

print "2.4 computePredictions:", ols.computePredictions(w,rdd_te)

print "2.5 computeError:", ols.computeError(w,rdd_te)
    
