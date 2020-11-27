#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:16:19 2020

@author: specialist
"""

# Import libraries
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import re
import string
import nltk
from nltk.corpus import words, stopwords
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql import SparkSession, functions as F 
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler, RegexTokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
import gensim.parsing.preprocessing as gsp
from pyspark.ml.feature import Word2Vec
import time

# Initialize english
stop_words = [i for i in stopwords.words('english')]

# Initialize spark
sc = SparkContext.getOrCreate(SparkConf().setMaster('local[*]'))
spark = SparkSession.builder.getOrCreate()

# Read data
train = spark.read.option('header', True).csv('/Users/specialist/Documents/data/tweets/train.csv')
test = spark.read.option('header', True).csv('/Users/specialist/Documents/data/tweets/test.csv')

# Inspect
print('Available Features- ')
print(train.columns)
print('Target Distribution: ')
train.groupby('target').count().show()

'''
    Text Preprocessing
'''

# Compile list of modifications
filters = [gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text]

# Function to apply preprocessing
def processing_pipeline(x):
    
    for f in filters:
        
        x = f(x)
        
    return x

proc_udf = udf(processing_pipeline, StringType())

# Apply function to process corpus
t0 = train.where(F.col('text').isNotNull())
t = t0.withColumn('txt', proc_udf(train.text))
 
# Fill missing values with 0
t = t.fillna({'target':'0'})

# Split data for training
(training, validation) = t.randomSplit([0.8, 0.2], seed = 100)

'''
    Machine Learning
'''

# Initialize model components
tokenizer = Tokenizer(inputCol = "txt", outputCol = "tkns")
vectorizer = Word2Vec(vectorSize = 500, minCount = 0, inputCol = "tkns", outputCol = "fts")
labs = StringIndexer(inputCol = 'target', outputCol = 'lbl')
nn = MultilayerPerceptronClassifier(labelCol = 'lbl', featuresCol = "fts", maxIter = 100, layers = [500, 250, 250, 2], blockSize = 128, seed = 100)    
pipe = Pipeline(stages = [tokenizer, vectorizer, labs, nn])

# Train and predict
preds = pipe.fit(training).transform(validation)

# Evaluate
scorer = BinaryClassificationEvaluator(labelCol = 'lbl')
print('Validation AUC for ROC of Embedding NN model is {}'.format(round(scorer.evaluate(preds), 3)))