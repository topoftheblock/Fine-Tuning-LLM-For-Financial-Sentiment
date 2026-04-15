import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset

# --- 1. Setup & Authentication ---
load_dotenv()
client = OpenAI()

# --- 2. Configuration ---
# Toggle these variables depending on whether you are generating 
# your Training set or your Validation set!

# For Training:   split="validation", TARGET_SAMPLES=500, output="train.jsonl"
# For Validation: split="train",      TARGET_SAMPLES=100, output="valid.jsonl"

DATA_SPLIT = "validation"  # Hugging Face dataset split to read from
TARGET_SAMPLES = 500       # How many rows to generate
OUTPUT_FILE = "./data/processed/train.jsonl" # Where to save the output

# --- 3. Download Raw Data ---
print(f"Fetching raw financial tweets from Hugging Face (Split: '{DATA_SPLIT}')...")
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split=DATA_SPLIT)
raw_texts = dataset['text'][:TARGET_SAMPLES]

# --- 4. Prepare the Output Directory ---
# Using './' ensures it stays inside your project folder!
os.makedirs("./data/processed", exist_ok=True)

print(f"🎓 Starting Teacher-Student Distillation on {TARGET_SAMPLES} samples...")
print(f" Saving progress in real-time to {OUTPUT_FILE}")

# Open the file in 'append' mode ('a'). 
# This prevents data loss if the script is interrupted or API times out.
with open(OUTPUT_FILE, "a") as f:
    for i, text in enumerate(raw_texts):
        
        # The Teacher Prompt (Telling GPT-4o-mini exactly how to behave)
        system_prompt = """You are an expert Wall Street quantitative analyst. 
        Read the financial text and output STRICTLY valid JSON with no markdown formatting.
        Format: {"ticker": "TICKER", "sentiment": [-1.0 to 1.0], "reasoning": "brief explanation"}"""
        
        try:
            # 1. Ask the Teacher (GPT-4o-mini)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0 # Strict deterministic output
            )
            
            # Parse the teacher's JSON response
            teacher_json = json.loads(response.choices[0].message.content.strip())
            
            # 2. Format for the Student (Qwen ChatML)
            qwen_instruction = (
                "Analyze the following text, extract the ticker, determine the sentiment "
                "(-1.0 to 1.0), and provide a brief reasoning. Output strictly in JSON format."
            )
            
            student_prompt = (
                f"<|im_start|>system\n{qwen_instruction}<|im_end|>\n"
                f"<|im_start|>user\nInput: {text}<|im_end|>\n"
                f"<|im_start|>assistant\n{json.dumps(teacher_json)}<|im_end|>\n"
            )
            
            # 3. Save to disk immediately
            f.write(json.dumps({"text": student_prompt}) + "\n")
            
            # Print a status update every 50 tweets
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{TARGET_SAMPLES} tweets...")
                
            # A tiny sleep to prevent hitting OpenAI API rate limits
            time.sleep(0.1)
            
        except Exception as e:
            # If a tweet has weird formatting or GPT breaks JSON, skip it gracefully
            print(f"Skipping row {i} due to error: {e}")

print(f"\n🎉 Distillation complete! Saved to {OUTPUT_FILE}")