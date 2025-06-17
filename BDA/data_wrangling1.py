# spark_data_wrangling.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, trim, regexp_replace


spark = SparkSession.builder \
    .appName("KindleReviewsDataWrangling") \
    .getOrCreate()

input_csv_path = "gs://shinchan12/kindle_reviews_single_csv/kindle_single_file.csv" # Ensure this points to the directory containing your single CSV file

output_transformed_path = "gs://shinchan12/kindle_reviews_transformed_csv"


print(f"Loading CSV data from: {input_csv_path}")

df = spark.read.csv(input_csv_path, header=True, inferSchema=True)

print("Mapping star ratings to sentiment labels and cleaning review_text...")
df_transformed = df.withColumn(
    "sentiment_label",
    when(col("overall_rating").between(1, 2), "Negative")
    .when(col("overall_rating") == 3, "Neutral")
    .when(col("overall_rating").between(4, 5), "Positive")
    .otherwise("Unknown") 
).withColumn(
    "review_text", 
    trim(lower(col("review_text").cast("string")))
)


final_columns = [
    "product_id",
    "reviewer_id",
    "reviewer_name", 
    "review_text",
    "overall_rating",
    "review_summary",
    "review_timestamp",
    "helpful_votes",
    "total_votes",
    "sentiment_label" 
]


df_final = df_transformed.select([col(c) for c in final_columns if c in df_transformed.columns])


print("Coalescing to 1 partition for single output file...")
df_single_partition = df_final.coalesce(1)


print(f"Saving transformed data to CSV at: {output_transformed_path}")
df_single_partition.write.option("header", True).csv(output_transformed_path, mode="overwrite")

print("Data wrangling and sentiment mapping complete!")


spark.stop()