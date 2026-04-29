import json
import time
import yfinance as yf
from confluent_kafka import Producer

# --- Configuration ---
KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'financial_raw_text'

TICKERS_TO_WATCH = ["AAPL", "TSLA", "NVDA", "AMD", "SPY", "MSFT", "AMZN", "META", "GOOGL"]

conf = {'bootstrap.servers': KAFKA_BROKER, 'client.id': 'yfinance_scraper'}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"❌ Delivery failed: {err}")

seen_news_ids = set()

print(f"🚀 Starting Yahoo Finance Live Streamer with RAG Context...")
print(f"📡 Sending breaking news to Kafka topic: {TOPIC_NAME}")

try:
    while True:
        for ticker in TICKERS_TO_WATCH:
            try:
                stock = yf.Ticker(ticker)
                
                # --- RAG: Fetch Current Market Trend ---
                try:
                    hist = stock.history(period="2d")
                    if len(hist) >= 2:
                        prev_close = hist['Close'].iloc[0]
                        current_price = hist['Close'].iloc[-1]
                        change_pct = ((current_price - prev_close) / prev_close) * 100
                        trend = f"Up {change_pct:.2f}% today" if change_pct >= 0 else f"Down {abs(change_pct):.2f}% today"
                    else:
                        trend = "Trend unavailable"
                except Exception:
                    trend = "Trend unavailable"

                # Fetch News
                news_items = stock.news
                
                for article in news_items:
                    article_id = article.get("uuid", "")
                    
                    if article_id and article_id not in seen_news_ids:
                        seen_news_ids.add(article_id)
                        
                        headline = article.get("title", "")
                        
                        # INJECT RAG CONTEXT INTO THE TEXT FIELD
                        formatted_text = f"${ticker} - {headline} | Current Market Context: {trend}"
                        
                        payload = {
                            "post_id": article_id[:8], 
                            "timestamp": int(article.get("providerPublishTime", time.time())),
                            "source": "Yahoo Finance",
                            "text": formatted_text,
                            "author": article.get("publisher", "Financial Reporter")
                        }

                        producer.produce(
                            topic=TOPIC_NAME, 
                            value=json.dumps(payload).encode('utf-8'), 
                            callback=delivery_report
                        )
                        producer.poll(0)
                        
                        print(f"📰 Sent: {formatted_text}")
                        
            except Exception as e:
                print(f"⚠️ Error fetching {ticker}: {e}")
                
        print("⏳ Waiting 60 seconds for new headlines...")
        time.sleep(60)

except KeyboardInterrupt:
    print("\n🛑 Stopping stream...")
finally:
    producer.flush()