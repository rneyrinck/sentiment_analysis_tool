import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from app.utils import validate_csv
from app.model import get_model, analyze_text, log_memory

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Input text must not be empty")


@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running on Render!"}


@app.post("/analyze")
def analyze_sentiment(input: TextInput):
    log_memory()  # Log memory usage before analysis
    model = get_model()
    result = analyze_text(model, input.text)
    log_memory()  # Log memory usage after analysis
    return result


@app.post("/batch")
async def batch_analyze(file: UploadFile = File(...)):
    try:
        data = validate_csv(file.file)
    except ValueError as e:
        return {"error": str(e)}  # Handle invalid CSV files

    results = []
    log_memory()  # Log memory usage before batch processing
    model = get_model()
    for row in data["text"]:
        try:
            result = analyze_text(model, row)
            results.append(result)
        except Exception as e:
            return {"error": f"Error processing row '{row}': {str(e)}"}
    log_memory()  # Log memory usage after batch processing
    return results


# Ensure correct port binding for Render
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
