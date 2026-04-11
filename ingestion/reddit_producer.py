import json
import time
import random
import uuid
from confluent_kafka import Producer

# --- Configuration ---
KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'financial_raw_text'

# Kafka Producer configuration
conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'client.id': 'reddit_scraper_v1'
}
producer = Producer(conf)

def delivery_report(err, msg):
    """Callback triggered on successful/failed message delivery."""
    if err is not None:
        print(f"❌ Message delivery failed: {err}")
    else:
        print(f"✅ Delivered to {msg.topic()} [{msg.partition()}]")

# --- Simulated Data Stream ---
# In production, you would replace this with the PRAW (Reddit API) live stream listener.
sample_posts = [
    "Just bought the dip on $AAPL, let's go!",
    "Is anyone else worried about the new interest rates?",  # No ticker, should be filtered
    "TSLA margins collapsed this quarter despite record deliveries. $TSLA",
    "I think $NVDA is overvalued at this point.",
    "Look at my cute dog!", # Spam, should be filtered
    "Just bought the dip on $AAPL, let's go!" # Duplicate, should be filtered
]

print(f"🚀 Starting Reddit Producer... streaming to {TOPIC_NAME}")

try:
    while True:
        # Create a mock social media payload
        payload = {
            "post_id": str(uuid.uuid4())[:8],
            "timestamp": int(time.time()),
            "source": "reddit/wallstreetbets",
            "text": random.choice(sample_posts),
            "author": f"user_{random.randint(100, 999)}"
        }

        # Convert to JSON and send to Kafka
        json_payload = json.dumps(payload)
        producer.produce(
            topic=TOPIC_NAME, 
            value=json_payload.encode('utf-8'), 
            callback=delivery_report
        )
        
        # Flush ensures it's sent immediately for our simulation
        producer.flush()
        
        # Wait a random amount of time between 0.5 and 2 seconds to simulate real streaming
        time.sleep(random.uniform(0.5, 2.0))

except KeyboardInterrupt:
    print("\n🛑 Stopping producer...")
finally:
    producer.flush()