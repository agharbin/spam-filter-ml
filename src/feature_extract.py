from __future__ import print_function
from pyspark import SparkContext, SparkConf 
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg import SparseVector
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import words
import sys
import pickle

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

def extract_words(text, dictionary):
    return filter_dictionary(lowercase(tokenize(text)), dictionary)

def calculate_tfidf(documents):
    hashingTF = HashingTF()
    tf = hashingTF.transform(documents.map(lambda x: x[1]))
    tf.cache()
    idf = IDF().fit(tf)
    return idf.transform(tf)

def raw_files_to_labeled_features(raw_files, label_file):
    # Initialize spark
    conf = SparkConf().setAppName("SpamFilter").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # Get the set of words that we will be accepting as valid features
    valid_words = set(w.lower() for w in words.words())

    # Load training data and convert to our desired format
    raw_files = sc.wholeTextFiles(raw_files)

    # Extract a document of filtered words from each text file
    documents = raw_files.map(lambda x: (x[0], extract_words(x[1], valid_words)))

    # Calculate TF-IDF values for each document
    tfidf = calculate_tfidf(documents)

    # Load labels
    labels = sc.parallelize(load_labels(label_file).map(lambda x: x[0])

    # Append indexes to features and labels
    indexed_labels = labels.zipWithIndex().map(lambda x: (x[1],x[0]))
    indexed_features = tfidf.zipWithIndex().map(lambda x: (x[1],x[0]))

    # Join labels and features into tuples and return
    return indexed_labels.join(indexed_features).map(lambda x: x[1]).collect()

if __name__ == "__main__":
    labeled_features = raw_files_to_labeled_features(sys.argv[1], sys.argv[2])

    output = open("./output.txt","w")
    pickle.dump(labeled, output)
