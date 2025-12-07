import json
import random
import time
from datetime import datetime
from kafka import KafkaProducer


producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_transaction():
      return {
        "transaction_id": random.randint(1_000_000, 9_999_999),
        "customer_id": random.randint(1, 10_000),
        "merchant_id": random.randint(1, 500),
        "amount": round(random.uniform(1, 1000), 2),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "is_fraud": random.choice([0, 0, 0, 0, 1])
    }

if __name__ == "__main__":
    print("Iniciando Kafka producer...")
    count = 0
    try:
        while True:
            event = generate_transaction()
            producer.send("transactions", value=event)
            count += 1
            if count % 100 == 0:
                print(f"Enviadas {count} transacciones")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\nTotal enviadas: {count}")