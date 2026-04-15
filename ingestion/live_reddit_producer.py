# ingestion/live_reddit_producer.py
import json
import time
import praw
from confluent_kafka import Producer

# --- Configuration ---
KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'financial_raw_text'

# Add your Reddit API credentials here
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="mac:finance.sentiment.bot:v1.0 (by /u/yourusername)"
)

# Kafka Producer
conf = {'bootstrap.servers': KAFKA_BROKER, 'client.id': 'live_reddit_scraper'}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"❌ Delivery failed: {err}")

print(f"🚀 Connecting to r/WallStreetBets live stream...")
print(f"📡 Sending data to Kafka topic: {TOPIC_NAME}")

try:
    # Stream live comments from WallStreetBets
    for comment in reddit.subreddit("wallstreetbets").stream.comments(skip_existing=True):
        
        # We only care about comments that mention a ticker (e.g., $AAPL, $TSLA)
        if "$" in comment.body:
            payload = {
                "post_id": str(comment.id),
                "timestamp": int(comment.created_utc),
                "source": "reddit/wallstreetbets",
                "text": comment.body.replace("\n", " ").strip(),
                "author": str(comment.author)
            }

            producer.produce(
                topic=TOPIC_NAME, 
                value=json.dumps(payload).encode('utf-8'), 
                callback=delivery_report
            )
            producer.poll(0) # Trigger delivery callbacks
            
            print(f"📥 Captured: {payload['text'][:80]}...")

except KeyboardInterrupt:
    print("\n🛑 Stopping stream...")
finally:
    producer.flush()