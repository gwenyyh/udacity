# import libraries
import datetime
from functools import reduce
import math
import matplotlib.pyplot as plt
import numpy as np

import pyspark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    isnan, 
    when, 
    count, 
    countDistinct, 
    avg, 
    col, 
    desc, 
    lit, 
    max, 
    split, 
    udf
)
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression, 
    GBTClassifier, 
    DecisionTreeClassifier, 
    NaiveBayes
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, 
    MulticlassClassificationEvaluator
)
from pyspark.ml.feature import (
    StandardScaler, 
    VectorAssembler, 
    UnivariateFeatureSelector
)
from pyspark.ml.tuning import (
    CrossValidator, 
    ParamGridBuilder, 
    CrossValidatorModel
)
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark_dist_explore import hist

import re
import seaborn as sns
