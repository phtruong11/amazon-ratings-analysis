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
    table = table.groupby(table.asin, table.year)\
        .agg(functions.avg(table.rate).alias("avg_rate"), functions.count(table.one).alias("sales")).sort(table.asin)
    list = table.groupby(table.asin).agg(functions.count(table.asin).alias('year_count'))
    list = list.where(list.year_count >= 3).select(list.asin)
    table = table.join(list, "asin").select(table.asin, table.year, table.avg_rate, table.sales)\
        .where(table.sales > 1000).sort(table.asin, table.year).coalesce(1)
    table = table.alias('this_year').join(table.alias('last_year'), 'asin')\
        .where(functions.col('this_year.year') == functions.col('last_year.year') + 1) \
        .select(functions.col('this_year.asin').alias('asin'),
            functions.col('this_year.year').alias('this_year'),
            functions.col('last_year.year').alias('last_year'),
            functions.col('this_year.avg_rate').alias('this_rate'),
            functions.col('last_year.avg_rate').alias('last_rate'),
            functions.col('this_year.sales').alias('this_sales'),
            functions.col('last_year.sales').alias('last_sales'))
    table = table.withColumn('rate_change', table.this_rate - table.last_rate)\
        .withColumn('sales_change', table.this_sales / table.last_sales - 1)
    table = table.select(table.asin, table.this_year, table.last_year,
                         functions.round(table.this_rate, scale=3).alias('rate'),
                         table.this_sales.alias('sales'),
                         functions.round(table.rate_change, scale=3).alias('rate_change'),
                         functions.round(table.sales_change, scale=3).alias('sales_change'))\
        .where((table.sales_change > 0.1) | (table.sales_change < -0.1))\
        .where((table.rate_change > 0.1) | (table.rate_change < -0.1))\
        .sort(table.sales_change, ascending=False).coalesce(1)
    total_table = table.withColumn('one', table.this_year ** 0)
    positive_avg = total_table.where(total_table.sales_change > 0)\
        .groupby().agg(functions.avg(total_table.rate)).collect()[0][0]
    negative_avg = total_table.where(total_table.sales_change < 0) \
        .groupby().agg(functions.avg(total_table.rate)).collect()[0][0]
    total = total_table.groupby().agg(functions.count(total_table.one)).collect()[0][0]
    both_increase = total_table.where((total_table.sales_change > 0) & (total_table.rate_change > 0))\
        .groupby().agg(functions.count(total_table.one)).collect()[0][0]
    both_decrease = total_table.where((total_table.sales_change < 0) & (total_table.rate_change < 0)) \
        .groupby().agg(functions.count(total_table.one)).collect()[0][0]
    print("total: {}, increase: {}, decrease: {}".format(total, both_increase, both_decrease))
    print("Both Increase: ", both_increase / total)
    print("Both Decrease: ", both_decrease / total)
    print("average of rates for positive sales: ", positive_avg)
    print("average of rates for negative sales: ", negative_avg)
    table.write.json(output, mode="overwrite")


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('wikipedia code').getOrCreate()
    assert spark.version >= '3.0'  # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)
