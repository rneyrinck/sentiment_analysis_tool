from transformers import pipeline
import os
import psutil

# Global model variable for lazy loading
model = None


def log_memory():
    """
    Logs the current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 ** 2  # Convert bytes to MB
    print(f"Memory usage: {memory_usage:.2f} MB")


def get_model():
    """
    Lazily loads the sentiment analysis model.
    Returns:
        The sentiment analysis pipeline.
    """
    global model
    if model is None:
        log_memory()  # Log memory usage before loading
        print("Loading the sentiment analysis model...")
        model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        log_memory()  # Log memory usage after loading
    return model


def analyze_text(model, text):
    """
    Analyze the sentiment of a single text input.

    Args:
        model: The loaded sentiment analysis model.
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary with the sentiment label and confidence score.
    """
    if not text.strip():  # Check for empty or whitespace-only input
        return {"sentiment": "neutral", "confidence": 0.0}

    result = model(text)[0]
    return {
        "sentiment": result["label"].lower(),
        "confidence": result["score"]
    }
