from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD, SVMWithSGD
import pickle
import sys

def toLabeledPoints(sc, data):
    return sc.parallelize(data).map(lambda x: LabeledPoint(x[0], x[1]))

def loadData(path):
    data_file = open(path,"r")
    return pickle.load(data_file)

def computeError(m, d):
    labelsAndPreds = d.map(lambda p: (p.label, m.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda (v,p): v != p).count() / float(d.count())
    return trainErr

if __name__ == "__main__":
    conf = SparkConf().setAppName("SpamFilter").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    data = toLabeledPoints(sc, loadData(sys.argv[1]))
    testData = toLabeledPoints(sc, loadData(sys.argv[2]))

    # Train the model with different regularization parameters
    results = []
    for i in [0.00, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        model = SVMWithSGD.train(data, step=0.05, regParam=i)
        results.append((i, computeError(model, testData)))

    outfile = open("results.txt","w")
    outfile.write(str(results))
    outfile.close()
