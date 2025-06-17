from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

# Start Spark session
spark = SparkSession.builder.appName("KindleJsonToCsvConversion").getOrCreate()

# Input and output paths
input_path = "gs://shinchan12/Kindle_Store.jsonl"
output_path = "gs://shinchan12/kindle_reviews_single_csv/"

# Read the JSON Lines file from GCS
df = spark.read.json(input_path)

# Select and rename columns as per requirements
df_selected = df.select(
    col("user_id").alias("reviewer_id"),
    col("asin").alias("product_id"),
    lit(None).cast("string").alias("reviewer_name"),
    col("text").alias("review_text"),
    col("rating").cast("double").alias("overall_rating"),
    col("title").alias("review_summary"),
    col("timestamp").cast("long").alias("review_timestamp"),
    col("helpful_vote").alias("helpful_votes"),
    lit(None).cast("long").alias("total_votes")
)

# Coalesce to a single partition for single CSV output
df_selected.coalesce(1).write.option("header", True).csv(output_path, mode="overwrite")

# Stop Spark session
spark.stop()
