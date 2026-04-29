import json
import os
import time
import random
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
client = OpenAI()

DATA_SPLIT = "train" 
TARGET_SAMPLES = 500
OUTPUT_FILE = "./data/processed/train_rag.jsonl"

dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split=DATA_SPLIT)
raw_texts = dataset['text'][:TARGET_SAMPLES]

os.makedirs("./data/processed", exist_ok=True)

print(f"🎓 Starting RAG Teacher-Student Distillation on {TARGET_SAMPLES} samples...")

# Generate a plausible simulated market trend for training variance
def generate_simulated_trend():
    direction = random.choice(["Up", "Down"])
    pct = round(random.uniform(0.1, 5.0), 2)
    return f"{direction} {pct}% today"

with open(OUTPUT_FILE, "a") as f:
    for i, text in enumerate(raw_texts):
        
        simulated_trend = generate_simulated_trend()
        # Inject context into the raw text just like the Kafka producer will
        rag_input_text = f"{text} | Current Market Context: {simulated_trend}"
        
        # Tell the Teacher to use the new Context
        system_prompt = """You are an expert Wall Street quantitative analyst. 
        Read the financial text and the provided market context. Calculate sentiment (-1.0 to 1.0).
        If the news is positive but the trend is heavily down (or vice versa), reflect that nuance in your reasoning.
        Output STRICTLY valid JSON with no markdown formatting.
        Format: {"ticker": "TICKER", "sentiment": 0.0, "reasoning": "brief explanation"}"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Input: {rag_input_text}"}
                ],
                temperature=0.0
            )
            
            teacher_json = json.loads(response.choices[0].message.content.strip())
            
            # Update Student Instructions
            qwen_instruction = (
                "Analyze the following text and its Current Market Context. "
                "Extract the ticker, determine the sentiment (-1.0 to 1.0) by weighing the news "
                "against the market trend, and provide a brief reasoning. Output strictly in JSON format."
            )
            
            student_prompt = (
                f"<|im_start|>system\n{qwen_instruction}<|im_end|>\n"
                f"<|im_start|>user\nInput: {rag_input_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{json.dumps(teacher_json)}<|im_end|>\n"
            )
            
            f.write(json.dumps({"text": student_prompt}) + "\n")
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{TARGET_SAMPLES} tweets...")
                
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Skipping row {i} due to error: {e}")

print(f"\n🎉 RAG Distillation complete! Saved to {OUTPUT_FILE}")