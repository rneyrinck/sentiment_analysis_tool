import pandas as pd
from fastapi import FastAPI
from fastapi import UploadFile, File
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")


# Define input schema
class TextInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running!"}


@app.post("/analyze")
def analyze_sentiment(input: TextInput):
    result = sentiment_model(input.text)[0]
    return {
        "sentiment": result["label"].lower(),
        "confidence": result["score"]
    }


@app.post("/batch")
async def analyze_batch(file: UploadFile = File(...)):
    # Read uploaded CSV
    data = pd.read_csv(file.file)
    if "text" not in data.columns:
        return {"error": "CSV must contain a 'text' column"}

    # Analyze sentiments
    results = [
        {"text": row, "sentiment": result["label"].lower(), "confidence": result["score"]}
        for row in data["text"]
        for result in [sentiment_model(row)]
    ]

    # Return results
    return results
