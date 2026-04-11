import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, regexp_extract, udf, from_unixtime
from pyspark.sql.types import StructType, StructField, StringType, LongType, FloatType

# --- 1. Initialize Spark Session ---
# We include both the Kafka and Elasticsearch connector packages
spark = SparkSession.builder \
    .appName("FinancialSignalFilter") \
    .config("spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.elasticsearch:elasticsearch-spark-30_2.12:8.12.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# --- 2. Define Schemas ---
# Schema for the incoming raw Kafka data
json_schema = StructType([
    StructField("post_id", StringType(), True),
    StructField("timestamp", LongType(), True),
    StructField("source", StringType(), True),
    StructField("text", StringType(), True),
    StructField("author", StringType(), True)
])

# Schema for the LLM API response
llm_response_schema = StructType([
    StructField("ticker", StringType(), True),
    StructField("sentiment", FloatType(), True),
    StructField("reasoning", StringType(), True)
])

# --- 3. Read Stream from Kafka ---
print("🌊 Spark Streaming Pipeline Started. Waiting for Kafka data...")
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "financial_raw_text") \
    .option("startingOffsets", "latest") \
    .load()

# Parse the JSON from Kafka's binary 'value' column
parsed_stream = raw_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), json_schema).alias("data")) \
    .select("data.*")

# --- 4. Filtering ---
# Extract tickers using Regex. Drops any message that doesn't explicitly mention a ticker like $AAPL
filtered_stream = parsed_stream \
    .withColumn("extracted_ticker", regexp_extract(col("text"), r"\$([A-Z]{1,5})", 1)) \
    .filter(col("extracted_ticker") != "")

# --- 5. The LLM Bridge (UDF) ---
def get_llm_sentiment(text):
    try:
        # Call your local M4 FastAPI Server
        # Note: Use localhost if running Spark natively on Mac. 
        # If running Spark inside Docker, change this to host.docker.internal
        res = requests.post("http://localhost:8000/analyze", json={"text": text}, timeout=10)
        
        if res.status_code == 200:
            data = res.json()
            return (
                data.get("ticker", "UNKNOWN"), 
                float(data.get("sentiment", 0.0)), 
                data.get("reasoning", "No reasoning provided.")
            )
    except Exception as e:
        # In a production environment, you would log this exception
        pass 
    
    return ("ERROR", 0.0, "API Failure")

# Register the Python function as a Spark UDF
llm_udf = udf(get_llm_sentiment, llm_response_schema)

# Apply the UDF to our filtered data stream to get the AI analysis
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

# --- 6. Format for Elasticsearch ---
# Convert the Unix timestamp (integer) to a proper Datetime format for Kibana charting
final_stream = enriched_stream \
    .withColumn("@timestamp", from_unixtime(col("timestamp")).cast("timestamp")) \
    .drop("timestamp")

# --- 7. Output Sink (Elasticsearch) ---
print("📊 Sending enriched AI signals to Elasticsearch...")

# Write the stream directly into the Elasticsearch index named 'financial_signals'
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