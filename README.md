# Fine-Tuning LLM for Financial Sentiment

An end-to-end real-time data pipeline for ingesting financial social media text, analyzing its sentiment using a fine-tuned Large Language Model (LLM) powered by Apple's MLX, and visualizing the results.

---

## Architecture Overview

The system is composed of four primary components.

**Data Ingestion** (`ingestion/reddit_producer.py`)

A Python script acting as a Kafka Producer simulates live financial posts from sources like WallStreetBets. It generates mock social media JSON payloads and streams them to a Kafka topic named `financial_raw_text`.

**LLM Inference Server** (`inference/server.py`)

A FastAPI server that hosts a fine-tuned Llama-3 model loaded into M4 Unified Memory via the `mlx_lm` library. It exposes a `POST /analyze` endpoint that accepts a text string and returns a strict JSON response containing the extracted ticker symbol, a sentiment score between -1.0 and 1.0, and the reasoning behind the score.

**Stream Processing** (`processing/spark_streaming.py`)

A PySpark streaming application that consumes raw text from the `financial_raw_text` Kafka topic. It uses regular expressions to filter out messages that do not mention a stock ticker (e.g., `$AAPL`), enriches the filtered stream by calling the LLM inference server via a Spark UDF, and writes the AI-enriched signals to an Elasticsearch index named `financial_signals`.

**Infrastructure** (`infrastructure/docker-compose.yml`)

Docker Compose manages the background services: a Bitnami Kafka broker configured with KRaft, a single-node Elasticsearch instance with security disabled for local development, and Kibana for dashboard visualization.

---

## Prerequisites

- **Python dependencies:** `mlx-lm`, `fastapi`, `uvicorn`, `pyspark`, `confluent-kafka`, `elasticsearch`
- **Model:** A fine-tuned Llama-3 model must be available at `./fused-finance-model` for the FastAPI server to load
- **Docker:** Docker and Docker Compose must be installed to run the Kafka and Elastic stack containers

---

## Getting Started

Follow these steps in order to run the full pipeline locally.

### 1. Start the Infrastructure

Navigate to the `infrastructure/` directory and bring up Kafka, Elasticsearch, and Kibana:

```bash
docker-compose up -d
```

### 2. Start the Inference Server

Boot the FastAPI server hosting the MLX Llama-3 model. It will be available at `localhost:8000`:

```bash
cd inference
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 3. Run the Spark Streaming Job

In a new terminal, launch the PySpark streaming job. The script automatically pulls in the required Spark packages for Kafka (`spark-sql-kafka-0-10`) and Elasticsearch (`elasticsearch-spark-30`):

```bash
python processing/spark_streaming.py
```

> **Note:** If you run Spark inside Docker rather than locally, update the inference URL in your `get_llm_sentiment` function from `http://localhost:8000/analyze` to `http://host.docker.internal:8000/analyze`.

### 4. Start the Data Producer

Start the flow of simulated Reddit data into Kafka:

```bash
python ingestion/reddit_producer.py
```

### 5. View the Dashboards

Open Kibana at `http://localhost:5601`. Create an index pattern for `financial_signals` to begin visualizing real-time AI sentiment analysis on stock tickers.

---

## Project Structure

```
.
├── ingestion/
│   └── reddit_producer.py       # Kafka producer simulating Reddit posts
├── inference/
│   └── server.py                # FastAPI LLM inference server
├── processing/
│   └── spark_streaming.py       # PySpark stream processor
└── infrastructure/
    └── docker-compose.yml       # Kafka, Elasticsearch, and Kibana services
```

---

## How It Works

1. The producer continuously emits mock financial posts to the `financial_raw_text` Kafka topic.
2. The Spark job consumes this stream and filters for messages containing a stock ticker symbol.
3. Each qualifying message is sent to the LLM inference server, which returns a structured sentiment payload.
4. The enriched records are indexed into Elasticsearch under `financial_signals`.
5. Kibana reads from that index, enabling real-time dashboard creation and visualization of sentiment trends by ticker.