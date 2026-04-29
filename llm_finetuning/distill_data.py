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
OUTPUT_FILE = "./data/processed/train_cot_multi.jsonl"

dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split=DATA_SPLIT)
raw_texts = dataset['text'][:TARGET_SAMPLES]

os.makedirs("./data/processed", exist_ok=True)

print(f"🎓 Starting CoT + Multi-Ticker Distillation on {TARGET_SAMPLES} samples...")

with open(OUTPUT_FILE, "a") as f:
    for i, text in enumerate(raw_texts):
        
        # The Teacher Prompt - Forcing CoT and Multi-Ticker Arrays
        system_prompt = """You are an expert Wall Street quantitative analyst. 
        Read the provided financial text. 
        
        TASK 1: If the text only mentions one company, creatively augment the text by adding a realistic sentence comparing it to a competitor (e.g., if AAPL is up, mention MSFT is lagging). 
        TASK 2: First, think step-by-step about the financial implications, the entities involved, and their respective sentiments. Wrap your reasoning strictly in <think>...</think> tags.
        TASK 3: Extract the tickers and sentiments (-1.0 to 1.0) for ALL companies mentioned. Output STRICTLY a JSON array of objects.
        
        Format exactly like this:
        [Augmented Text]
        <think>
        Step-by-step reasoning here...
        </think>
        [
            {"ticker": "TICKER1", "sentiment": 0.8, "reasoning": "brief explanation"},
            {"ticker": "TICKER2", "sentiment": -0.5, "reasoning": "brief explanation"}
        ]"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Input: {text}"}
                ],
                temperature=0.2 # Slight temperature to allow creative text augmentation
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Extract the augmented text, think block, and JSON array from the Teacher's response
            # (Assuming the teacher follows the format perfectly)
            parts = raw_response.split("<think>")
            augmented_text = parts[0].replace("[Augmented Text]", "").strip()
            rest = "<think>" + parts[1]
            
            # The Student Instruction (What Qwen will see during inference)
            qwen_instruction = (
                "Analyze the following text. First, provide step-by-step reasoning inside <think>...</think> tags. "
                "Then, extract the tickers, sentiments (-1.0 to 1.0), and reasoning for ALL companies mentioned "
                "as a strict JSON array of objects."
            )
            
            student_prompt = (
                f"<|im_start|>system\n{qwen_instruction}<|im_end|>\n"
                f"<|im_start|>user\nInput: {augmented_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{rest}<|im_end|>\n"
            )
            
            f.write(json.dumps({"text": student_prompt}) + "\n")
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{TARGET_SAMPLES} tweets...")
                
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Skipping row {i} due to error or formatting issue: {e}")

print(f"\n🎉 CoT Distillation complete! Saved to {OUTPUT_FILE}")