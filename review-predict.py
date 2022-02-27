import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
import re, string
from pyspark.ml.feature import VectorAssembler, SQLTransformer, Binarizer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# from sklearn import metrics

review_schema = types.StructType([
    types.StructField('reviewerID', types.StringType()),
    types.StructField('asin', types.StringType()),
    types.StructField('reviewerName', types.StringType()),
    types.StructField('helpful', types.ArrayType(types.IntegerType())),
    types.StructField('reviewText', types.StringType()),
    types.StructField('overall', types.FloatType()),
    types.StructField('summary', types.StringType()),
    types.StructField('unixReviewTime', types.IntegerType()),
    types.StructField('reviewTime', types.StringType())
])

wordsep = re.compile(r'[%s\s]+' % re.escape(string.punctuation))

@functions.udf(returnType=types.IntegerType())
def count_word(w):
    word = wordsep.split(w)
    return len(word)

@functions.udf(returnType=types.IntegerType())
def count_sentence(s):
    sentence = re.split(r'[!?]+|(?<!\.)\.(?!\.)',s)
    return len(sentence)

def create_data(input):
    data = spark.read.json(input, schema=review_schema)
    data = data.select('helpful','reviewText','overall')
    data = data.withColumn('helpful_votes', data['helpful'][0])
    data = data.withColumn('total_votes', data['helpful'][1])
    data = data.filter(data['total_votes'] > 10)
    data = data.withColumn('helpful', data['helpful_votes']/data['total_votes'])
    data = data.withColumn('review_len', functions.length('reviewText'))
    data = data.withColumn('word_count', count_word(data['reviewText']))
    data = data.withColumn('sentence_count', count_sentence(data['reviewText']))
    #data.show(10)
    training, validation = data.randomSplit([0.75, 0.25])
    return training, validation

def classification(input):
    training, validation = create_data(input)

    binarizer = Binarizer(inputCol='helpful', outputCol='helpful_sign', threshold=0.6)
    assemble_features = VectorAssembler(inputCols=['overall', 'review_len', 'word_count', 'sentence_count'], outputCol='features')
    classifier = GBTClassifier(featuresCol='features', labelCol='helpful_sign', maxIter=100, maxDepth=5)
    pipeline = Pipeline(stages=[binarizer, assemble_features, classifier])

    model = pipeline.fit(training)
    predictions = model.transform(validation)
    predictions.show()

    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='helpful_sign')
    score = evaluator.evaluate(predictions)
    print(score)

    # y_true = predictions.select(['helpful_sign']).collect()
    # y_pred = predictions.select(['prediction']).collect()
    # print("Confusion Matrix: ", metrics.confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    spark = SparkSession.builder.appName('Review Prediction').getOrCreate()
    assert spark.version >= '3.0'
    spark.sparkContext.setLogLevel('WARN')

    input = sys.argv[1]
    classification(input)
