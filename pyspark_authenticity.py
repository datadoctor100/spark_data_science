#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:40:35 2020

@author: specialist
"""

# Import libraries
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.sql.functions import isnan, when, count, col, lit, concat, udf, sum, mean
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import re
import string
import nltk
from nltk.corpus import stopwords
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import CountVectorizer
from textblob import TextBlob
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, ArrayType, FloatType
from pyspark.ml.clustering import LDA
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pyspark.mllib.evaluation import BinaryClassificationMetrics

stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())

# Initialize spark
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.getOrCreate()

# Read data
train = spark.read.options(header = True, index = True).csv('/Users/specialist/Documents/Data/authtrain.csv')
test = spark.read.options(header = True, index = True).csv('/Users/specialist/Documents/Data/authtest.csv')

print('Available features- ')
print(train.columns)

train.select('text').show(5, truncate = False)

# Drop nulls
t = train[~train.text.isNull()]

'''
    Text processing
'''

# Remove punctuation and digits
puncs = re.compile('[%s]' % re.escape(string.punctuation))
nums = re.compile('(\\d+)')

puncsudf = udf(lambda x: re.sub(puncs,' ', x).lower())
numsudf = udf(lambda x: re.sub(nums,' ', x).strip().replace('  ', ' '))

t1 = t.withColumn('txt', puncsudf('text'))
t1.select('txt').show(5, truncate = False)

t2 = t1.withColumn('txt', numsudf('txt'))
t2.select('txt').show(5, truncate = False)

# Function to process
def prepare_txt(x):
    
    data = [i for i in x.split() if i not in stop_words]
    tags = nltk.pos_tag(data)
    x1 = [x[0] for x in tags if x[0] in words and len(x[0]) > 3 and x[1].startswith(('N', 'J', 'V'))]

    if len(x1) > 0:
        
        return ' '.join(x1)
    
    else:
        
        return None

text_prepper = udf(lambda x: prepare_txt(x), StringType())

# Apply function
t3 = t2.withColumn('pos', text_prepper('txt'))
t3.select('pos').show(15, truncate = False)

# Filter empty tweets
t4 = t3[~t3.pos.isNull()] 
t4.select('pos').show(15, truncate = False)

'''
    Classification
'''

t4.groupBy('target').count().show()

# Fill null with authentic
t5 = t4.fillna({'target':1})
t5.groupBy('target').count().show()
t6 = t5.withColumn('label', col('target').cast(IntegerType()))

# Split the data 
(train, val) = t6.randomSplit([0.8, 0.2], 100)

# Initialize components
tokenizer = Tokenizer(inputCol = "pos", outputCol = "wrds")
hasher = HashingTF(inputCol = 'wrds', outputCol = "raw")
idf = IDF(minDocFreq = 3, inputCol = "raw", outputCol = "features")
nb = NaiveBayes()
rf = RandomForestClassifier(seed = 100)
gbt = GBTClassifier()

# Fit NB pipeline
pipeline = Pipeline(stages = [tokenizer, hasher, idf, nb])
model = pipeline.fit(train)
preds = model.transform(val)
evaluator = BinaryClassificationEvaluator()
print("AUC for Naive Bayes Model is %s.." % str(evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})))

# Fit RF pipeline
pipeline = Pipeline(stages = [tokenizer, hasher, idf, rf])
model = pipeline.fit(train)
preds = model.transform(val)
evaluator = BinaryClassificationEvaluator()
print("AUC for RF Model is %s.." % str(evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})))

# Fit GBT pipeline
pipeline = Pipeline(stages = [tokenizer, hasher, idf, gbt])
model = pipeline.fit(train)
preds = model.transform(val)
evaluator = BinaryClassificationEvaluator()
print("AUC for GBT Model is %s.." % str(evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})))

