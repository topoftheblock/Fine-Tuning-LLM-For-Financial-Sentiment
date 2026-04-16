# ingestion/yahoo_finance_producer.py
import json
import time
import yfinance as yf
from confluent_kafka import Producer

# --- Configuration ---
KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'financial_raw_text'

# The stocks you want to monitor for breaking news
TICKERS_TO_WATCH = ["AAPL", "TSLA", "NVDA", "AMD", "SPY", "MSFT", "AMZN", "META", "GOOGL"]

# Kafka Producer setup
conf = {'bootstrap.servers': KAFKA_BROKER, 'client.id': 'yfinance_scraper'}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"❌ Delivery failed: {err}")

# Keep track of news we have already sent so we don't send duplicates to Kafka
seen_news_ids = set()

print(f"🚀 Starting Yahoo Finance Live Streamer...")
print(f"📡 Sending breaking news to Kafka topic: {TOPIC_NAME}")

try:
    while True:
        for ticker in TICKERS_TO_WATCH:
            try:
                # Fetch data from Yahoo Finance (No API Key needed!)
                stock = yf.Ticker(ticker)
                news_items = stock.news
                
                for article in news_items:
                    # Yahoo Finance usually provides a UUID for each article
                    article_id = article.get("uuid", "")
                    
                    if article_id and article_id not in seen_news_ids:
                        seen_news_ids.add(article_id)
                        
                        headline = article.get("title", "")
                        
                        # FORMATTING TRICK: We add "$TICKER" to the front of the headline.
                        # Your Spark script uses regex to find the stock, so we must include the $ symbol!
                        formatted_text = f"${ticker} - {headline}"
                        
                        # Build the exact JSON payload Spark is expecting
                        payload = {
                            "post_id": article_id[:8], 
                            "timestamp": int(article.get("providerPublishTime", time.time())),
                            "source": "Yahoo Finance",
                            "text": formatted_text,
                            "author": article.get("publisher", "Financial Reporter")
                        }

                        # Send to Kafka
                        producer.produce(
                            topic=TOPIC_NAME, 
                            value=json.dumps(payload).encode('utf-8'), 
                            callback=delivery_report
                        )
                        producer.poll(0)
                        
                        print(f"📰 Sent: {formatted_text}")
                        
            except Exception as e:
                print(f"⚠️ Error fetching {ticker}: {e}")
                
        # Wait 60 seconds before polling Yahoo Finance again so we don't get IP banned
        print("⏳ Waiting 60 seconds for new headlines...")
        time.sleep(60)

except KeyboardInterrupt:
    print("\n🛑 Stopping stream...")
finally:
    producer.flush()