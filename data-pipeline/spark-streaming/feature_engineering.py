from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def build_features(df: DataFrame) -> DataFrame:
    df = df.withColumn("event_time", F.to_timestamp("timestamp"))

    windowed = (
        df
        .withWatermark("event_time", "2 minutes")
        .groupBy(
            F.window("event_time", "60 seconds").alias("w"),
            F.col("customer_id")
        )
        .agg(
            F.count("*").alias("transaction_count_1m"),
            F.avg("amount").alias("avg_amount_1m"),
            F.max("amount").alias("max_amount_1m"),
            F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)).alias("fraud_count_1m"),
        )
    )

    result = (
        windowed
        .withColumn("window_start", F.col("w").start)
        .withColumn("window_end", F.col("w").end)
        .drop("w")
    )

    return result
