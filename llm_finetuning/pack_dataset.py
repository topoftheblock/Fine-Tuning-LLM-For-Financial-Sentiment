import json
from transformers import AutoTokenizer

# Load the Llama-3 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

MAX_TOKENS = 2048 # A safe context window size for M4 training

# Load your standard training data
with open("../data/processed/train.jsonl", "r") as f:
    raw_data = [json.loads(line)["text"] for line in f.readlines()]

packed_dataset = []
current_block = ""
current_tokens = 0

print(f"📦 Packing {len(raw_data)} individual tweets into high-density training blocks...")

for text in raw_data:
    # Count how many tokens this single tweet uses
    text_tokens = len(tokenizer.encode(text))
    
    # If adding this tweet exceeds our max block size, save the current block and start a new one
    if current_tokens + text_tokens > MAX_TOKENS:
        packed_dataset.append({"text": current_block})
        current_block = text + "\n"
        current_tokens = text_tokens
    else:
        # Otherwise, append it to the current block
        current_block += text + "\n"
        current_tokens += text_tokens

# Don't forget the last block
if current_block:
    packed_dataset.append({"text": current_block})

# Save the packed dataset
with open("../data/processed/packed_train.jsonl", "w") as f:
    for item in packed_dataset:
        f.write(json.dumps(item) + "\n")

print(f"✅ Compressed into {len(packed_dataset)} high-density blocks.")
print("Update your train_mlx_v2.sh to point to 'packed_train.jsonl' for 3x faster training!")