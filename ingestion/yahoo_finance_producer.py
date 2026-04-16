import json
import time
import random
import uuid
from confluent_kafka import Producer

KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'financial_raw_text'

producer = Producer({'bootstrap.servers': KAFKA_BROKER, 'client.id': 'sim_wsb'})

tickers = ["$AAPL", "$TSLA", "$NVDA", "$AMD", "$SPY"]
bullish_phrases = [
    "calls are going to print tomorrow!", "is breaking out, buying the dip.",
    "margins are fat, to the moon 🚀", "earnings are gonna crush estimates.",
    "literally cannot go down."
]
bearish_phrases = [
    "puts are locked in, market is bleeding.", "CEO just sold shares, get out now.",
    "inflation is destroying this.", "is a massive bubble waiting to pop.",
    "guidance was terrible, dropping this."
]

print(f"🚀 Starting High-Volume WSB Simulator... streaming to {TOPIC_NAME}")

try:
    while True:
        ticker = random.choice(tickers)
        if random.random() > 0.5:
            text = f"{ticker} {random.choice(bullish_phrases)}"
        else:
            text = f"{ticker} {random.choice(bearish_phrases)}"

        payload = {
            "post_id": str(uuid.uuid4())[:8],
            "timestamp": int(time.time()),
            "source": "reddit/wallstreetbets",
            "text": text,
            "author": f"WSB_degen_{random.randint(100, 999)}"
        }

        producer.produce(TOPIC_NAME, value=json.dumps(payload).encode('utf-8'))
        producer.poll(0)
        
        print(f"🔥 Sent: {text}")
        time.sleep(random.uniform(0.5, 2.0)) 

except KeyboardInterrupt:
    print("\n🛑 Stopping...")
finally:
    producer.flush()