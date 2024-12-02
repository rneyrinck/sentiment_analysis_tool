from fastapi import FastAPI
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
