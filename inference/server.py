import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlx_lm import load, generate

app = FastAPI(title="M4 Financial Sentiment API (RAG-Enabled)")

print("Loading fused Qwen 0.5B model into M4 Unified Memory...")
MODEL_PATH = "./fused-qwen-finance" 
model, tokenizer = load(MODEL_PATH)
print("✅ Qwen Model loaded and API is ready!")

class AnalysisRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(request: AnalysisRequest):
    # UPDATED SYSTEM PROMPT to explicitly tell the model to use the context
    system_instruction = (
        "Analyze the following text and its Current Market Context. "
        "Extract the ticker, determine the sentiment (-1.0 to 1.0) by weighing the news "
        "against the market trend, and provide a brief reasoning. Output strictly in JSON format."
    )
    
    prompt = (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\nInput: {request.text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        response_text = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=150, 
            verbose=False
        )
        
        clean_json = json.loads(response_text.strip())
        
        return {
            "ticker": clean_json.get("ticker", "UNKNOWN"),
            "sentiment": clean_json.get("sentiment", 0.0),
            "reasoning": clean_json.get("reasoning", "No reasoning provided.")
        }

    except json.JSONDecodeError:
        return {
            "ticker": "UNKNOWN", 
            "sentiment": 0.0, 
            "reasoning": f"LLM Format Error. Raw output: {response_text}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))