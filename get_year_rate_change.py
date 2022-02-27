import sys

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
import json
from pyspark.sql import SparkSession, functions, types
import os


@functions.udf(returnType=types.IntegerType())
def unix_time_to_year(unix_time):
    return int(unix_time / 365 / 24 / 60 / 60 + 1 + 1970)


def main(inputs, output):
    # main logic starts here
    schema = types.StructType([
        types.StructField('asin', types.StringType()),
        types.StructField('reviewer', types.StringType()),
        types.StructField('rate', types.DoubleType()),
        types.StructField('unix_time', types.LongType())
    ])
    table = spark.read.csv(inputs, schema=schema, sep=',')
    table = table.withColumn("year", unix_time_to_year(table.unix_time)).withColumn("one", table.rate ** 0)
    table = table.groupby(table.year).agg(functions.avg(table.rate).alias("avg_rate"), functions.count(table.one)
                                          .alias("sales")).sort(table.year)
    table = table.alias('this_year').join(table.alias('last_year'), functions.col('this_year.year') == functions.col('last_year.year') + 1, 'left')\
        .select(functions.col('this_year.year').alias('this_year'),
                functions.col('last_year.year').alias('last_year'),
                functions.col('this_year.avg_rate').alias('this_rate'),
                functions.col('last_year.avg_rate').alias('last_rate'),
                functions.col('this_year.sales').alias('this_sales'),
                functions.col('last_year.sales').alias('last_sales'))
    table = table.withColumn('rate_change', table.this_rate - table.last_rate) \
        .withColumn('sales_change', table.this_sales / table.last_sales - 1)
    table = table.select(table.this_year.alias('year'),
                         functions.round(table.this_rate, scale=3).alias('avg_rate'),
                         table.this_sales.alias('sales'),
                         functions.round(table.rate_change, scale=3).alias('rate_change'),
                         functions.round(table.sales_change, scale=3).alias('sales_change'))
    table = table.sort(table.year).where(table.sales > 10000).coalesce(1)
    table.write.json(output, mode="overwrite")


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('wikipedia code').getOrCreate()
    assert spark.version >= '3.0'  # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)
