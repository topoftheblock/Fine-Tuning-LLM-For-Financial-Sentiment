import json
import os
from openai import OpenAI

# Initialize OpenAI Client (Ensure your OPENAI_API_KEY is set in your terminal)
client = OpenAI()

# Simulated raw data (In reality, load this from a CSV or Kafka dump)
raw_unlabelled_tweets = [
    "Inflation printed at 2%, paving the way for aggressive rate cuts. Bullish for $SPY.",
    "CEO of $TSLA just sold another 5 billion in stock to fund his other companies.",
    "The new regulatory probe into the banking sector looks like a nothing-burger."
]

distilled_dataset = []

print("🎓 Starting Teacher-Student Distillation using GPT-4o-mini...")

for tweet in raw_unlabelled_tweets:
    # 1. The Teacher Prompt
    # We explicitly tell GPT-4o exactly how to act and format the output
    system_prompt = """You are an expert Wall Street quantitative analyst. 
    Read the financial text and output STRICTLY valid JSON with no markdown formatting.
    Format: {"ticker": "TICKER", "sentiment": [-1.0 to 1.0], "reasoning": "brief explanation"}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": tweet}
            ],
            temperature=0.0 # Deterministic output
        )
        
        teacher_json = json.loads(response.choices[0].message.content.strip())
        
        # 2. Format it for the Student (Llama-3 on MLX)
        student_prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Analyze the following text, extract the ticker, determine the sentiment "
            "(-1.0 to 1.0), and provide a brief reasoning. Output strictly in JSON format.\n\n"
            f"Input: {tweet}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{json.dumps(teacher_json)}<|eot_id|>"
        )
        
        distilled_dataset.append({"text": student_prompt})
        print(f"✅ Distilled: {tweet[:40]}... -> {teacher_json['sentiment']}")
        
    except Exception as e:
        print(f"❌ Failed to distill tweet: {e}")

# 3. Save the new high-quality synthetic data
os.makedirs("../data/processed", exist_ok=True)
with open("../data/processed/synthetic_train.jsonl", "w") as f:
    for item in distilled_dataset:
        f.write(json.dumps(item) + "\n")

print("\n🎉 Distillation complete! Saved to data/processed/synthetic_train.jsonl")