

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("KindleReviewsSentimentClassification") \
    .getOrCreate()

# --- Configuration ---
# Input path for your transformed data (from Task 3)
input_transformed_path = "gs://shinchan12/processed_data/cleaned_kindle.csv"

# Path to save the trained ML model
model_output_path = "gs://shinchan12/models1/sentiment_lr_model_sample"

# Path to save the generated plot to your GCS bucket
plot_output_path = "gs://shinchan12/plots1/confusion.png"

# --- 1. Load Data ---
print(f"Loading transformed data from: {input_transformed_path}")
df = spark.read.csv(input_transformed_path, header=True, inferSchema=True)

# Drop rows where 'review_text' or 'sentiment_label' is null, as they are crucial for training
print("Dropping rows with null review_text or sentiment_label...")
df = df.dropna(subset=["review_text", "sentiment_label"])

# Ensure 'review_text' is string and 'overall_rating' is double
print("Casting review_text to string and overall_rating to double...")
df = df.withColumn("review_text", col("review_text").cast("string")) \
       .withColumn("overall_rating", col("overall_rating").cast("double"))


sample_fraction = 0.25 # <--- ADJUST THIS BASED ON YOUR SOURCE FILE SIZE AND TARGET 2.5 GB
print(f"Sampling data to approximately {sample_fraction * 100}% of the original size for processing...")
df = df.sample(False, sample_fraction, seed=42) # False for no replacement
print(f"Sampled DataFrame has {df.count()} rows.")

# --- 2. Prepare Data for MLlib ---
print("Indexing sentiment labels...")
label_indexer = StringIndexer(inputCol="sentiment_label", outputCol="indexedLabel").fit(df)
df_indexed = label_indexer.transform(df)

# --- 3. Feature Engineering (Text Processing Pipeline) ---
print("Tokenizing review text...")
tokenizer = Tokenizer(inputCol="review_text", outputCol="words")

print("Removing stop words...")
stopwords_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered_words")

print("Hashing features...")
hashing_tf = HashingTF(inputCol=stopwords_remover.getOutputCol(), outputCol="raw_features", numFeatures=10000)

print("Applying IDF...")
idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol="features")

text_processing_pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf])

# --- 4. Prepare for Model Training ---
final_df = df_indexed.withColumnRenamed("indexedLabel", "label")

# --- 5. Split Data into Training and Test Sets ---
print("Splitting data into training and test sets (80/20)...")
(trainingData, testData) = final_df.randomSplit([0.8, 0.2], seed=42)

# --- MANUAL OVERSAMPLING FOR CLASS IMBALANCE (Since classWeight is not available) ---
print("Performing manual oversampling for class imbalance in training data...")
class_counts = trainingData.groupBy("label").count().collect()

# Determine the count of the majority class
majority_class_count = 0
for row in class_counts:
    if row['count'] > majority_class_count:
        majority_class_count = row['count']

resampled_training_dfs = []
for row in class_counts:
    class_label = row['label']
    class_count = row['count']
    
    # Calculate the ratio needed to bring minority classes up to majority class count
    if class_count > 0 and class_count < majority_class_count:
        # We need to sample with replacement `sampling_ratio` times
        sampling_ratio = float(majority_class_count) / class_count
        print(f"  Class {class_label} (actual count: {class_count}) needs oversampling by factor {sampling_ratio:.2f}")
        # Sample with replacement to increase the count
        resampled_df_for_class = trainingData.filter(col("label") == class_label).sample(True, sampling_ratio, seed=42)
        resampled_training_dfs.append(resampled_df_for_class)
    else:
        # For the majority class or already balanced classes, just add them
        print(f"  Class {class_label} (actual count: {class_count}) - no oversampling needed.")
        resampled_training_dfs.append(trainingData.filter(col("label") == class_label))

if resampled_training_dfs:
    # Union all resampled DataFrames
    trainingData_rebalanced = resampled_training_dfs[0]
    for i in range(1, len(resampled_training_dfs)):
        trainingData_rebalanced = trainingData_rebalanced.unionAll(resampled_training_dfs[i])
    trainingData = trainingData_rebalanced # Use the rebalanced data for training
    print(f"Training data rebalanced. New total rows: {trainingData.count()}")
else:
    print("Warning: No data available to rebalance. Training will proceed with original data.")

# --- End of Manual Oversampling ---

# --- 6. Choose and Train a Machine Learning Model (Logistic Regression) ---
print("Training Logistic Regression model...")
# classWeight REMOVED as it's not supported in older Spark versions
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

ml_pipeline = Pipeline(stages=[text_processing_pipeline, lr])

# Train with the rebalanced trainingData
model = ml_pipeline.fit(trainingData)

# --- 7. Make Predictions on Test Data ---
print("Making predictions on the test set...")
predictions = model.transform(testData)
predictions.cache()

# --- 8. Model Evaluation ---
print("Evaluating model performance...")
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy = {accuracy}")

f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)
print(f"Test F1 Score = {f1_score}")

precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = precision_evaluator.evaluate(predictions)
print(f"Test Precision = {precision}")

recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = recall_evaluator.evaluate(predictions)
print(f"Test Recall = {recall}")

# --- 9. Visualize Model Performance (Confusion Matrix) ---
print("Generating Confusion Matrix visualization...")

target_sample_size_for_plot = 100000

if predictions.count() > target_sample_size_for_plot:
    sample_fraction_for_plot = float(target_sample_size_for_plot) / predictions.count()
    pandas_predictions = predictions.select("label", "prediction").sample(False, sample_fraction_for_plot, seed=42).toPandas()
    print(f"Sampled {pandas_predictions.shape[0]} predictions for visualization.")
else:
    pandas_predictions = predictions.select("label", "prediction").toPandas()
    print(f"Collected all {pandas_predictions.shape[0]} predictions for visualization.")

original_labels = label_indexer.labels
sorted_numeric_labels = list(range(len(original_labels)))

cm = confusion_matrix(pandas_predictions['label'], pandas_predictions['prediction'], labels=sorted_numeric_labels)

cm_df = pd.DataFrame(cm, index=[f'Actual: {l}' for l in original_labels],
                     columns=[f'Predicted: {l}' for l in original_labels])

plt.figure(figsize=(9, 7))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False, linewidths=.5, linecolor='black')
plt.title('Confusion Matrix for Kindle Reviews Sentiment Classification (Sampled Data)')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()

local_plot_filename = "/tmp/confusion.png"
plt.savefig(local_plot_filename)
print(f"Confusion Matrix plot generated locally at: {local_plot_filename}")

try:
    gcs_output_dir = os.path.dirname(plot_output_path)
    jvm_uri = spark.sparkContext._jvm.java.net.URI
    jvm_conf = spark.sparkContext._jvm.org.apache.hadoop.conf.Configuration()
    jvm_fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem
    jvm_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path

    jvm_fs.get(jvm_uri(gcs_output_dir), jvm_conf).mkdirs(jvm_path(gcs_output_dir))

    jvm_fs.get(jvm_uri(plot_output_path), jvm_conf).copyFromLocalFile(
        jvm_path(local_plot_filename),
        jvm_path(plot_output_path)
    )
    print(f"Confusion Matrix plot copied to GCS: {plot_output_path}")
except Exception as e:
    print(f"Error copying plot to GCS: {e}")
    print("You might need to manually copy it using 'gsutil cp' from the master node if this fails:")
    print(f"  gsutil cp {local_plot_filename} {plot_output_path}")

if os.path.exists(local_plot_filename):
    os.remove(local_plot_filename)
    print(f"Cleaned up local temporary file: {local_plot_filename}")

# --- 10. Save the Trained Model ---
model.write().overwrite().save(model_output_path)
print(f"Trained model saved to: {model_output_path}")

# Stop Spark Session
spark.stop()