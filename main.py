from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os
import sys

# Import preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.preprocess import clean_text

app = FastAPI(title="Vietnamese Fake News Detection API")

# Allow CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = os.path.join("models", "fake_news_model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("Warning: Model not found. Please train the model first.")
except Exception as e:
    print(f"Error loading model: {e}")

class NewsItem(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(item: NewsItem):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    cleaned_text = clean_text(item.text)
    prediction = model.predict([cleaned_text])[0]
    probability = model.predict_proba([cleaned_text]).max()
    
    label = "Fake" if prediction == 1 else "Real"
    
    return {
        "text": item.text,
        "prediction": label,
        "confidence": float(probability)
    }

# Serve Frontend (after build)
# We will assume 'frontend/dist' exists for production
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)