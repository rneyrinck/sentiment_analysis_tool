from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from transformers import pipeline
from app.utils import validate_csv

# Initialize FastAPI app
app = FastAPI()

# Load sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")


# Define input schema
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Input text must not be empty")


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
async def batch_analyze(file: UploadFile = File(...)):
    try:
        data = validate_csv(file.file)
    except ValueError as e:
        return {"error": str(e)}  # Handle invalid CSV files

    results = []
    for row in data["text"]:
        # print(f"row: {row}")
        try:
            result = sentiment_model(row)  # Get prediction
            # print(f"sentiment_model output for '{row}': {result}")
            results.append({
                "text": row,
                "sentiment": result[0]["label"].lower(),
                "confidence": result[0]["score"]
            })
        except Exception as e:
            # print(f"Error processing row '{row}': {str(e)}")
            return {"error": f"Error processing row '{row}': {str(e)}"}

    return results
