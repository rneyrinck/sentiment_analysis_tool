from fastapi import FastAPI, UploadFile, File

from app.model import load_model, analyze_text, analyze_batch
from app.utils import preprocess_text, validate_csv

# Initialize FastAPI app and model
app = FastAPI()
model = load_model()


@app.post("/analyze")
def analyze(input_text: str):
    # Preprocess input text
    clean_text = preprocess_text(input_text)
    # Analyze sentiment
    result = analyze_text(model, clean_text)
    return result


@app.post("/batch")
async def batch_analyze(file: UploadFile = File(...)):
    # Validate CSV
    data = validate_csv(file.file)
    # Preprocess and analyze
    results = analyze_batch(model, data["text"].tolist())
    return results
