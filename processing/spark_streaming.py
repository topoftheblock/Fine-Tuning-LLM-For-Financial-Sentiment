import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, regexp_extract, udf, from_unixtime
from pyspark.sql.types import StructType, StructField, StringType, LongType, FloatType

# UPDATED: Added spark.jars.excludes to prevent the Elasticsearch connector
# from pulling in conflicting Spark 3.3.x dependencies that break PySpark 3.5.0
spark = SparkSession.builder \
    .appName("FinancialSignalFilter") \
    .config("spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.elasticsearch:elasticsearch-spark-30_2.12:8.12.0") \
    .config("spark.jars.excludes", 
            "org.apache.spark:spark-yarn_2.12,org.apache.spark:spark-core_2.12,org.apache.spark:spark-sql_2.12,org.apache.spark:spark-catalyst_2.12") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

json_schema = StructType([
    StructField("post_id", StringType(), True),
    StructField("timestamp", LongType(), True),
    StructField("source", StringType(), True),
    StructField("text", StringType(), True),
    StructField("author", StringType(), True)
])

llm_response_schema = StructType([
    StructField("ticker", StringType(), True),
    StructField("sentiment", FloatType(), True),
    StructField("reasoning", StringType(), True)
])

print("🌊 Spark Streaming Pipeline Started. Waiting for Kafka data...")
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "financial_raw_text") \
    .option("startingOffsets", "latest") \
    .load()

parsed_stream = raw_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), json_schema).alias("data")) \
    .select("data.*")

filtered_stream = parsed_stream \
    .withColumn("extracted_ticker", regexp_extract(col("text"), r"\$([A-Z]{1,5})", 1)) \
    .filter(col("extracted_ticker") != "")

def get_llm_sentiment(text):
    try:
        res = requests.post("http://localhost:8000/analyze", json={"text": text}, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return (
                data.get("ticker", "UNKNOWN"), 
                float(data.get("sentiment", 0.0)), 
                data.get("reasoning", "No reasoning provided.")
            )
    except Exception as e:
        pass 
    return ("ERROR", 0.0, "API Failure")

llm_udf = udf(get_llm_sentiment, llm_response_schema)

enriched_stream = filtered_stream \
    .withColumn("llm_analysis", llm_udf(col("text"))) \
    .select(
        col("timestamp"),
        col("source"),
        col("text"),
        col("llm_analysis.ticker").alias("llm_ticker"),
        col("llm_analysis.sentiment").alias("sentiment_score"),
        col("llm_analysis.reasoning").alias("llm_reasoning")
    )

final_stream = enriched_stream \
    .withColumn("@timestamp", from_unixtime(col("timestamp")).cast("timestamp")) \
    .drop("timestamp")

print("📊 Sending enriched AI signals to Elasticsearch...")

query = final_stream.writeStream \
    .outputMode("append") \
    .format("org.elasticsearch.spark.sql") \
    .option("checkpointLocation", "/tmp/spark_checkpoints") \
    .option("es.nodes", "localhost") \
    .option("es.port", "9200") \
    .option("es.resource", "financial_signals") \
    .option("es.index.auto.create", "true") \
    .start()

query.awaitTermination()