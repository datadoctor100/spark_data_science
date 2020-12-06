#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 06:57:49 2020

@author: specialist
"""

# Import libraries
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, ArrayType, FloatType
from pyspark.sql import SparkSession
import re
import string
from pyspark.sql.functions import udf, col, size
import nltk
from nltk.corpus import stopwords
import pyspark.sql.functions as F
from pyspark.sql.functions import lit
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import CountVectorizer
from textblob import TextBlob

stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())

# Initialize spark
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.getOrCreate()

# Read data
df = spark.read.options(header = True).csv('/Users/specialist/Documents/data/hashtag_donaldtrump.csv')
df1 = spark.read.options(header = True).csv('/Users/specialist/Documents/data/hashtag_joebiden.csv')

# Inspect
df.columns
df.count()
df1.count()

# Label
df = df.withColumn('label', lit(0))
df1 = df1.withColumn('label', lit(1))

# Concat
tweets = df.union(df1)
tweets.select('tweet').show(5, truncate = False)

# Drop nulls
t = tweets[~tweets.tweet.isNull()]

'''
    Text processing
'''

# Remove punctuation and digits
puncs = re.compile('[%s]' % re.escape(string.punctuation))
nums = re.compile('(\\d+)')

puncsudf = udf(lambda x: re.sub(puncs,' ', x))
numsudf = udf(lambda x: re.sub(nums,' ', x).lower().split())

t1 = t.withColumn('txt', puncsudf('tweet'))
t2 = t1.withColumn('txt', numsudf('txt'))

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
t3 = t2.withColumn('pos', text_prepper('txt'))
t3.select('pos').show(15, truncate = False)

# Filter empty tweets
t4 = t3[~t3.pos.isNull()] 
t4.select('pos').show(15, truncate = False)

'''
    Sentiment Analysis
'''

# Function to determine sentiment
def get_sentiment(x):
    
    return TextBlob(x).sentiment.polarity

sentimentudf = udf(get_sentiment , FloatType())

# Apply function to get sentiment
t5 = t4.withColumn('sentiment', sentimentudf(t4.pos))

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
t6 = t5.withColumn('tone', toneudf(t5.sentiment))

t7 = t6.sample(.2, 100)
t7.count()

'''
    Classification
'''

# Split the data 
(train, val) = t7.randomSplit([0.8, 0.2], 100)

# Initialize components
tokenizer = Tokenizer(inputCol = "pos", outputCol = "wrds")
hasher = HashingTF(inputCol = 'wrds', outputCol = "raw")
idf = IDF(minDocFreq = 3, inputCol = "raw", outputCol = "features")
nb = NaiveBayes()
rf = RandomForestClassifier(seed = 100)

# Fit NB pipeline
pipeline = Pipeline(stages = [tokenizer, hasher, idf, nb])
model = pipeline.fit(train)
preds = model.transform(val)
metrics = BinaryClassificationMetrics(preds)
print("AUC for Naive Bayes Model is %s.." % metrics.areaUnderROC)

# Fit RF pipeline
pipeline = Pipeline(stages = [tokenizer, hasher, idf, rf])
model = pipeline.fit(train)
predictions = model.transform(val)
print("AUC for Random Forest Model is %s.." % metrics.areaUnderROC)

'''
    Topic Modeling
'''

# Initialize components
vectorizer = CountVectorizer(inputCol = 'wrds', outoutCol = 'raw')
lda = LDA(k = 6, seed = 100, optimizer = 'em')
pipeline = Pipeline(stages = [tokenizer, vectorizer, idf, lda])

# Build
model = pipeline.fit(t7)

# Function to get terms
def get_terms_udf(vocabulary):
    
    def termsIdx2Term(termIndices):
        
        return [vocabulary[int(index)] for index in termIndices]
    
    return udf(termsIdx2Term, ArrayType(StringType()))

# Get terms
vectors = model.stages[1]
vocab = vectors.vocabulary
final = t7.withColumn("Terms", get_terms_udf(vocab)("termIndices"))