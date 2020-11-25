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
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# Initialize english
english = set(words.words())
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

# Subset data w/ labels
t0 = train.filter(train.target.isNotNull())

'''
    Text Processing-
        1) Remove hyperlinks
        2) Remove punctuation
        3) Remove digits
        4) Remove stopwords
        5) Check english
'''

# Remove hyperlinks
t1 = t0.withColumn("txt", F.regexp_replace(col('text'), r"http\S+", '')) 

# Remove punctuation                 
t2 = t1.withColumn('txt', F.regexp_replace(col('txt'), '[^\sa-zA-Z0-9]', ''))

# Remove digits
t3 = t2.withColumn('txt', F.regexp_replace(col('txt'), '\d+', ''))

# Function to remove stopwords
def rm_stopwords(x):
    
    return ' '.join([i.lower() for i in x.split(' ') if i.lower() not in stop_words])

stop_words_udf = udf(rm_stopwords, StringType())

# Function to remove nonenglish
def check_english(x):
    
    return ' '.join([i.lower() for i in x.split(' ') if i.lower() not in english])

english_udf = udf(check_english, StringType())

# Apply function to remove stopwords
t4 = t3.withColumn('txt', stop_words_udf(t3.txt))

# Apply function to remove nonenglish
t5 = t4.withColumn('txt', english_udf(t4.txt))

# Inspect
t5.select("txt").show(5, truncate = False)

'''
    NLP
'''

# Tokenize
tokenizer = Tokenizer(inputCol = "txt", outputCol = "wrds")
wrds = tokenizer.transform(t5)

# TF-IDF
hasher = HashingTF(inputCol = "wrds", outputCol = "raw", numFeatures = 20)
fts = hasher.transform(wrds)

idf = IDF(inputCol = "raw", outputCol = "features")
idfm = idf.fit(fts)

out = idfm.transform(fts)

'''
    Machine Learning - RF
'''

# Split for training / testing
train, val = out.randomSplit([.7, .3])

# Initialize components
lblsidx = StringIndexer(inputCol = "target", outputCol = "target_idx").fit(out)
ftsidx = VectorIndexer(inputCol = "features", outputCol= "fts_idx").fit(out)

rf = RandomForestClassifier(labelCol = "target_idx", featuresCol = "fts_idx")

converter = IndexToString(inputCol = "prediction", outputCol = "target_pred", labels = lblsidx.labels)

# Assemble pipeline
pipeline = Pipeline(stages = [lblsidx, ftsidx, rf, converter])

# Train 
model = pipeline.fit(train)

# Predict
preds = model.transform(val)
scorer = BinaryClassificationEvaluator(labelCol = 'target_idx')

print('Validation AUC for ROC of RF model is {}'.format(round(scorer.evaluate(preds), 3)))

'''
    Machine Learning - GBT
'''

gbt = GBTClassifier(labelCol = 'target_idx', featuresCol = 'fts_idx')
gbtp = Pipeline(stages = [lblsidx, ftsidx, gbt, converter])
gbtm = gbtp.fit(train)
gbtpreds = gbtm.transform(val)
print('Validation AUC for ROC of GBT model is {}'.format(round(scorer.evaluate(gbtpreds), 3)))

