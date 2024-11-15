import logging
from time import time
from loguru import logger


def setup_logging(log_path="logs/runtime_logs.log"):
    """
    Set up logging for the application using Loguru.
    
    Args:
        log_path (str): Path to save log files.
    """
    logger.add(log_path, rotation="10 MB", retention="7 days", compression="zip")
    logger.info("Logging setup complete")


def timing(func):
    """
    Decorator to measure the execution time of a function.
    
    Args:
        func (function): Function to measure.
    
    Returns:
        wrapper: Wrapped function.
    """
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def chunked_processing(data, chunk_size):
    """
    Generator to split data into chunks for processing.
    
    Args:
        data (iterable): Input data.
        chunk_size (int): Size of each chunk.
    
    Yields:
        list: Chunks of the input data.
    """
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


@timing
def read_parquet_in_chunks(spark, path, chunk_size):
    """
    Read a large Parquet file in chunks using Spark.
    
    Args:
        spark (SparkSession): Spark session.
        path (str): Path to the Parquet file.
        chunk_size (int): Number of rows per chunk.
    
    Yields:
        DataFrame: Chunked Spark DataFrame.
    """
    df = spark.read.parquet(path)
    total_rows = df.count()
    for i in range(0, total_rows, chunk_size):
        yield df.limit(chunk_size).offset(i)
