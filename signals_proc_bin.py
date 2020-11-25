#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 06:41:36 2020

@author: specialist
"""

# Import libraries
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
#from pyspark.ml.clustering import Kmeans
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import GBTClassifier

# Initialize spark
sc = SparkContext.getOrCreate(SparkConf().setMaster('local[*]'))
spark = SparkSession.builder.getOrCreate()

# Read data
df = spark.read.load('/Users/specialist/Documents/data/a2.parquet')

# Create view
df.createOrReplaceTempView('tbl')

# Inspect
spark.sql('select * from tbl limit 5').show()

# Get target dist
spark.sql("select count(class), class from tbl group by class").show()
#df.groupby('CLASS').count().show()

# Initalize ML components
assembler = VectorAssembler(inputCols = ['X', 'Y', 'Z'], outputCol = 'fts')
pipeline = Pipeline(stages = [assembler])
pmod = pipeline.fit(df)
df1 = pmod.transform(df)

# Split data for training
train, test = df1.randomSplit([0.7, 0.3], seed = 100)

'''
     Model - Logistic Regression
'''

lr = LogisticRegression(featuresCol = 'fts', labelCol = 'CLASS', maxIter = 10)
lrm = lr.fit(train)
preds = lrm.transform(test)

# Evaluate
evaluator = BinaryClassificationEvaluator(labelCol = 'CLASS')
print('AUC for Logistic Regression Model ROC on Test Set: ', evaluator.evaluate(preds))

'''
    Model - Decision Tree
'''

dt = DecisionTreeClassifier(featuresCol = 'fts', labelCol = 'CLASS')
dtree = dt.fit(train)
dpreds = dtree.transform(test)

print('AUC for Decision Tree Model ROC on Test Set: ', evaluator.evaluate(dpreds))

'''
    Model - Gradient Boosted Tree
'''

gbt = GBTClassifier(featuresCol = 'fts', labelCol = 'CLASS', maxIter = 10)
gbtm = gbt.fit(train)
gbtpreds = gbtm.transform(test)

print('AUC for Decision Tree Model ROC on Test Set: ', evaluator.evaluate(gbtpreds))
