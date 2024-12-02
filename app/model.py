from transformers import pipeline


# Load the pre-trained sentiment analysis model
def load_model():
    """
    Load and return the sentiment analysis model from Hugging Face Transformers.
    """
    return pipeline("sentiment-analysis")


# Function to analyze a single text input
def analyze_text(model, text):
    """
    Analyze the sentiment of a single text input.

    Args:
        model: The loaded sentiment analysis model.
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary with the sentiment label and confidence score.
    """
    result = model(text)[0]
    return {
        "sentiment": result["label"].lower(),
        "confidence": result["score"]
    }


# Function to analyze a batch of texts
def analyze_batch(model, texts):
    """
    Analyze the sentiment of a batch of text inputs.

    Args:
        model: The loaded sentiment analysis model.
        texts (list of str): A list of text inputs to analyze.

    Returns:
        list of dict: A list of dictionaries with sentiment labels and confidence scores.
    """
    results = model(texts)
    return [
        {"text": text, "sentiment": res["label"].lower(), "confidence": res["score"]}
        for text, res in zip(texts, results)
    ]
