import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from feature_engineering import build_features

KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "transactions"

POSTGRES_URL = "jdbc:postgresql://localhost:5432/ml_data"
POSTGRES_TABLE = "transaction_features"
POSTGRES_USER = "mluser"
POSTGRES_PASSWORD = "mlpass"


def write_to_postgres(batch_df, batch_id):
    (
        batch_df
        .write
        .format("jdbc")
        .mode("append")
        .option("driver", "org.postgresql.Driver")
        .option("url", POSTGRES_URL)
        .option("dbtable", POSTGRES_TABLE)
        .option("user", POSTGRES_USER)
        .option("password", POSTGRES_PASSWORD)
        .save()
    )


def main():
    spark = (
        SparkSession.builder
        .appName("FraudFeatureStreaming")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    raw_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    value_df = raw_df.selectExpr("CAST(value AS STRING) as json_str")

    parsed_df = value_df.select(
        F.from_json(
            "json_str",
            """
            struct<
                transaction_id:bigint,
                customer_id:int,
                merchant_id:int,
                amount:double,
                timestamp:string,
                is_fraud:int
            >
            """
        ).alias("data")
    ).select("data.*")

    features_df = build_features(parsed_df)


    query = (
        features_df.writeStream
        .outputMode("append")
        .foreachBatch(write_to_postgres)
        .option("checkpointLocation", "/tmp/checkpoints/features_to_pg")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
