import json
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

print("🔍 Loading V1 Model to mine hard negatives...")
model, tokenizer = load("../inference/fused-finance-model")
strict_sampler = make_sampler(temp=0.0)

# Load your validation dataset (the data the model hasn't trained on)
with open("../data/processed/valid.jsonl", "r") as f:
    validation_data = [json.loads(line) for line in f.readlines()]

hard_negatives = []

print(f"Testing {len(validation_data)} examples to find weaknesses...")

for item in validation_data:
    # Split the training string to isolate just the user prompt
    parts = item["text"].split("<|start_header_id|>assistant<|end_header_id|>\n\n")
    prompt = parts[0] + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Extract the true sentiment from the ground truth
    true_answer_str = parts[1].replace("<|eot_id|>", "")
    true_sentiment = json.loads(true_answer_str)["sentiment"]
    
    # Let the model guess
    response = generate(model, tokenizer, prompt=prompt, max_tokens=150, sampler=strict_sampler, verbose=False)
    
    failed = False
    try:
        predicted_sentiment = json.loads(response.strip())["sentiment"]
        # If the model's guess is off by more than 0.5 (e.g., predicted 0.8, true was -0.2)
        if abs(predicted_sentiment - true_sentiment) > 0.5:
            failed = True
    except json.JSONDecodeError:
        failed = True # Breaking JSON is an automatic failure

    if failed:
        # Save the full, CORRECT example into our hard negatives pile
        hard_negatives.append(item)
        print("🚩 Found a Hard Negative!")

# Save the hard negatives so you can append them to your V2 training run
with open("../data/processed/hard_negatives.jsonl", "w") as f:
    for item in hard_negatives:
        f.write(json.dumps(item) + "\n")

print(f"\n🎯 Mined {len(hard_negatives)} hard negatives. Add these back to your training data!")