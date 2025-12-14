import time
import random
import uuid
from dataclasses import dataclass
from typing import Literal

import requests

API_URL = "http://192.168.139.2/predict"


@dataclass
class CustomerProfile:
    id: str
    risk_level: Literal["normal", "risky"]


def generate_features(profile: CustomerProfile) -> dict:
    """Generate realistic 1-minute aggregate features for a customer."""
    if profile.risk_level == "normal":
        transaction_count_1m = random.randint(0, 5)
        avg_amount_1m = round(random.uniform(5, 150), 2)
        max_amount_1m = round(avg_amount_1m * random.uniform(1.0, 3.0), 2)
        fraud_count_1m = 0
    else:  # risky
        transaction_count_1m = random.randint(5, 20)
        avg_amount_1m = round(random.uniform(100, 1000), 2)
        max_amount_1m = round(avg_amount_1m * random.uniform(1.5, 5.0), 2)
        fraud_count_1m = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]

    return {
        "transaction_count_1m": transaction_count_1m,
        "avg_amount_1m": avg_amount_1m,
        "max_amount_1m": max_amount_1m,
        "fraud_count_1m": fraud_count_1m,
    }


def main():
    # Create a small pool of customers
    customers = []
    for _ in range(8):
        customers.append(CustomerProfile(id=str(uuid.uuid4())[:8], risk_level="normal"))
    for _ in range(2):
        customers.append(CustomerProfile(id=str(uuid.uuid4())[:8], risk_level="risky"))

    print(f"Starting traffic generator against {API_URL}")
    print("Press Ctrl+C to stop.\n")

    while True:
        profile = random.choice(customers)
        features = generate_features(profile)

        payload = {
            "amount": round(random.uniform(5, 500), 2),
            "customer_id": int(profile.id[:6], 16) % 1_000_000,
            "country": random.choice(["ES", "FR", "DE", "US"]),
            **features,
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=2)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] request failed: {e}")
            time.sleep(1.0)
            continue

        label = "FRAUD" if data["is_fraud"] == 1 else "OK"
        print(
            f"[{profile.id}][{profile.risk_level}] "
            f"tx_count={payload['transaction_count_1m']}, "
            f"avg={payload['avg_amount_1m']}, "
            f"max={payload['max_amount_1m']}, "
            f"fraud_1m={payload['fraud_count_1m']} "
            f"=> {label} (p={data['fraud_probability']:.3f}, "
            f"latency={data['latency_ms']:.1f} ms)"
        )

        # Sleep a bit to simulate streaming traffic
        time.sleep(random.uniform(0.3, 1.5))


if __name__ == "__main__":
    main()
