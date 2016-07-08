from __future__ import print_function
from pyspark import SparkContext, SparkConf 
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg import SparseVector
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import words
import sys
import pickle

DATA_FILE_EXT = ".eml"
EXTENSION_LEN = 4

def load_labels(label_path):
    label_file = open(label_path, "r")
    labels = [line.strip().split() for line in label_file]
    return labels

def tokenize(text):
    return word_tokenize(BeautifulSoup(text).get_text())

def lowercase(text):
    return map(lambda x: x.lower(), text)

def filter_dictionary(text, dictionary):
    return filter(lambda x: x in dictionary, text)

def feature_extract(text, dictionary):
    return filter_dictionary(lowercase(tokenize(text)), dictionary)

if __name__ == "__main__":
    # Initialize spark
    conf = SparkConf().setAppName("SpamFilter").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # Get the set of words that we will be accepting as valid features
    valid_words = set(w.lower() for w in words.words())

    # Load training data and convert to our desired format
    raw_files = sc.wholeTextFiles(sys.argv[1])
    words = raw_files.map(lambda x: (x[0], map(lambda x: x.lower(),word_tokenize(BeautifulSoup(x[1]).get_text()))))

    # Filter by the words we want to use as features
    words_filtered = words.map(lambda x: (x[0],filter(lambda y: y in valid_words,x[1])))
    #words_filtered = raw_files.map(lambda x: (x[0], feature_extract(x[1], valid_words)))

    # Calculate TF-IDF values for each document
    hashingTF = HashingTF()
    tf = hashingTF.transform(words_filtered.map(lambda x: x[1]))
    tf.cache()
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)

    # Add a constant feature for the weighting factor
    #features = tfidf.map(lambda x: SparseVector(x.size + 1, map(lambda y: y + 1,x.indices), x.values))

    #features2 = features.map(lambda x: SparseVector(x.size, [0] + x.indices, [1.0] + x.values))

    labels = sc.parallelize(load_labels(sys.argv[2])).map(lambda x: x[0])

    i_labels = labels.zipWithIndex().map(lambda x: (x[1],x[0]))

    #i_features = features2.zipWithIndex().map(lambda x: (x[1],x[0]))
    i_features = tfidf.zipWithIndex().map(lambda x: (x[1],x[0]))

    labeled_features = i_labels.join(i_features).map(lambda x: x[1]).collect()

    #for i in range(10):
    #    print(labeled_features[i])

    output = open("./output.txt","w")
    pickle.dump(labeled_features, output)
