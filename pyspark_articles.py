#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 07:19:02 2020

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
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import re
import string
import nltk
from nltk.corpus import stopwords
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import CountVectorizer
from textblob import TextBlob
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, ArrayType, FloatType
from pyspark.ml.clustering import LDA

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())

# Initialize spark
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.getOrCreate()

# Read data
df = spark.read.options(header = True, index = True).csv('/Users/specialist/Documents/Data/articles_data.csv')

print('Available features- ')
print(df.columns)

# Combine text columns
df1 = df.withColumn('txt', concat(col("title"), lit(" "), col("description")))
df2 = df1.withColumn('txt', concat(col("txt"), lit(" "), col("content")))

# Drop nulls
df3 = df2[~df2.txt.isNull()]

'''
    Text processing
'''

# Remove punctuation and digits
puncs = re.compile('[%s]' % re.escape(string.punctuation))
nums = re.compile('(\\d+)')

puncsudf = udf(lambda x: re.sub(puncs,' ', x))
numsudf = udf(lambda x: re.sub(nums,' ', x).lower().split())

df4 = df3.withColumn('txt', puncsudf('txt'))
df5 = df4.withColumn('txt', numsudf('txt'))

# Function to process
def prepare_txt(x):
    
    data = [i for i in x if i not in stop_words]
    tags = nltk.pos_tag(data)
    x1 = [x[0] for x in tags if x[0] in words and len(x[0]) > 3 and x[1].startswith(('N', 'J', 'V'))]

    if len(x1) > 0:
        
        return ' '.join(x1)
    
    else:
        
        return None

text_prepper = udf(lambda x: prepare_txt(x), StringType())

# Apply function
df6 = df5.withColumn('pos', text_prepper('txt'))

# Filter empty tweets
df6 = df6[~df6.pos.isNull()] 

df6.select('pos').show(15, truncate = False)

'''
    Sentiment
'''

# Function to determine sentiment
def get_sentiment(x):
    
    return TextBlob(' '.join(x)).sentiment.polarity

sentimentudf = udf(get_sentiment , FloatType())

# Apply function to get sentiment
df7 = df6.withColumn('sentiment', sentimentudf(df6.txt))

# Classify sentiment
def get_tone(score):
    
    if (score >= 0.1):
        
        label = "positive"
   
    elif (score <= -0.1):
        
        label = "negative"
        
    else:
        
        label = "neutral"
        
    return label

toneudf = udf(get_tone, StringType())

# Apply function to get tone
df7 = df7.withColumn('tone', toneudf(df7.sentiment))

'''
    Topics
'''

# Initialize components
tokenizer = Tokenizer(inputCol = "pos", outputCol = "wrds")
cv = CountVectorizer(inputCol = 'wrds', outputCol = "raw")
idf = IDF(minDocFreq = 3, inputCol = "raw", outputCol = "features")
lda = LDA(k = 2, seed = 100, optimizer = 'em')
pipeline = Pipeline(stages = [tokenizer, cv, idf])

# Build
model = pipeline.fit(df7)
df8 = model.transform(df7)
mod = lda.fit(df8)
topics = mod.describeTopics(5)
    
# Function to get terms
def get_terms_udf(vocabulary):
    
    def get_terms_udf(termIndices):
        
        return [vocabulary[int(index)] for index in termIndices]
    
    return udf(get_terms_udf, ArrayType(StringType()))

# Get terms
vecmod = model.stages[1]
vocab = vecmod.vocabulary
final = topics.withColumn("terms", get_terms_udf(vocab)("termIndices"))
