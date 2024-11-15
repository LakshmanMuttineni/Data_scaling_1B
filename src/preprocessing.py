from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
import os


def initialize_spark(app_name="Preprocessing", master="local[*]"):
    """
    Initialize a Spark session.
    
    Args:
        app_name (str): Name of the Spark application.
        master (str): Master URL for the cluster.
    
    Returns:
        SparkSession: Initialized Spark session.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.sql.execution.arrow.enabled", "true") \
        .getOrCreate()


def load_dataset(spark, input_path):
    """
    Load the dataset using Spark.
    
    Args:
        spark (SparkSession): Spark session.
        input_path (str): Path to the input dataset (CSV).
    
    Returns:
        DataFrame: Spark DataFrame containing the dataset.
    """
    return spark.read.csv(input_path, header=True, inferSchema=True)


def scale_to_1b_records(df, target_size=1_000_000_000):
    """
    Scale a Spark DataFrame to 1 billion rows.
    
    Args:
        df (DataFrame): Input Spark DataFrame.
        target_size (int): Target number of rows.
    
    Returns:
        DataFrame: Scaled Spark DataFrame.
    """
    current_size = df.count()
    multiplier = target_size // current_size
    extra_rows = target_size % current_size

    # Repeat dataset `multiplier` times
    replicated_df = df
    for _ in range(multiplier - 1):
        replicated_df = replicated_df.union(df)
    
    # Add `extra_rows` by sampling from the original dataset
    extra_df = df.sample(withReplacement=True, fraction=extra_rows / current_size)
    return replicated_df.union(extra_df)


def save_to_parquet(df, output_path):
    """
    Save a Spark DataFrame to Parquet format.
    
    Args:
        df (DataFrame): Spark DataFrame to save.
        output_path (str): Path to save the Parquet file.
    """
    df.write.mode("overwrite").parquet(output_path)


if __name__ == "__main__":
    # Paths
    INPUT_PATH = "data/raw/Credit.csv"  # Adjust path as needed
    OUTPUT_PATH = "data/scaled/credit_1b.parquet"

    # Initialize Spark session
    spark = initialize_spark()

    # Load dataset
    print("Loading dataset...")
    df = load_dataset(spark, INPUT_PATH)

    # Scale dataset to 1 billion rows
    print("Scaling dataset to 1 billion rows...")
    scaled_df = scale_to_1b_records(df)

    # Save scaled dataset
    print("Saving scaled dataset to Parquet format...")
    save_to_parquet(scaled_df, OUTPUT_PATH)

    print(f"Dataset saved to {OUTPUT_PATH}")
