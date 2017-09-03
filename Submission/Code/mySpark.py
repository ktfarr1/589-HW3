from itertools import groupby

class RDD:
    def __init__(self,x):
        '''
        Create an RDD object from the given iterable. 
        '''
        self.data = list(x)
    
    def map(self, f):
        '''
        Return a new RDD by applying a function to each element of this RDD.
    
            >>> rdd = sc.parallelize(["b", "a", "c"])
            >>> sorted(rdd.map(lambda x: (x, 1)).collect())
            [('a', 1), ('b', 1), ('c', 1)]
        '''
        return(RDD(map(f, self.data)))   
        
    def reduce(self, f):
        '''
        Reduces the elements of this RDD using the specified commutative and
            associative binary operator. Currently reduces partitions locally.
    
            >>> from operator import add
            >>> sc.parallelize([1, 2, 3, 4, 5]).reduce(add)
            15
            >>> sc.parallelize((2 for _ in range(10))).map(lambda x: 1).cache().reduce(add)
            10
            >>> sc.parallelize([]).reduce(add)
            Traceback (most recent call last):
                ...
            ValueError: Can not reduce() empty RDD
        '''
        return(reduce(f, self.data))    
        
    def collect(self):
        '''Return a list that contains all of the elements in this RDD.    
           .. note:: This method should only be used if the resulting array is expected
                to be small, as all the data is loaded into the driver's memory.
        '''
        return(self.data)    
        
    def filter(self,f):
        '''
        Return a new RDD containing only the elements that satisfy a predicate.
    
            >>> rdd = sc.parallelize([1, 2, 3, 4, 5])
            >>> rdd.filter(lambda x: x % 2 == 0).collect()
            [2, 4]
        '''
        return(RDD(filter(f,self.data)))    

    def reduceByKey(self, f):
        '''
        Merge the values for each key using an associative and commutative reduce function.
    
        This will also perform the merging locally on each mapper before
        sending results to a reducer, similarly to a "combiner" in MapReduce.
        
        >>> from operator import add
        >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
        >>> sorted(rdd.reduceByKey(add).collect())
        [('a', 2), ('b', 1)]
        '''
        
        get_first = lambda p: p[0]
        get_second = lambda p: p[1]
        return(RDD(map(
            lambda l: (l[0], reduce(f, map(get_second, l[1]))),
            groupby(sorted(self.data, key=get_first), get_first)
        )))
        
    def take(self,K=10):
        '''
        Take the first num elements of the RDD.
    
        .. note:: this method should only be used if the resulting array is expected
            to be small, as all the data is loaded into the driver's memory.
    
        >>> sc.parallelize([2, 3, 4, 5, 6]).cache().take(2)
        [2, 3]
        >>> sc.parallelize([2, 3, 4, 5, 6]).take(10)
        [2, 3, 4, 5, 6]
        >>> sc.parallelize(range(100), 100).filter(lambda x: x > 90).take(3)
        [91, 92, 93]
        '''
        return(self.data[:min(len(self.data),K)])    

class SparkContext:
    
    def parallelize(self, x):
        '''
        Distribute a local Python collection to form an RDD. Using xrange
            is recommended if the input represents a range for performance.
    
            >>> sc.parallelize([0, 2, 3, 4, 6], 5).glom().collect()
            [[0], [2], [3], [4], [6]]
            >>> sc.parallelize(xrange(0, 6, 2), 5).glom().collect()
            [[], [0], [], [2], [4]]
        '''     
        return(RDD(x))
      
    def textFile(self, name):
        '''
        Read a text file from HDFS, a local file system (available on all
        nodes), or any Hadoop-supported file system URI, and return it as an
        RDD of Strings.
    
        >>> path = os.path.join(tempdir, "sample-text.txt")
        >>> with open(path, "w") as testFile:
        ...    _ = testFile.write("Hello world!")
        >>> textFile = sc.textFile(path)
        >>> textFile.collect()
        [u'Hello world!']
        '''    
        text_file = open(name, "r")
        lines = text_file.readlines()
        return(RDD(lines))
      