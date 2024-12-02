import pandas as pd


# Function to preprocess text (if needed)
def preprocess_text(text):
    """
    Preprocess the input text for sentiment analysis.

    Args:
        text (str): The raw text input.

    Returns:
        str: The cleaned text.
    """
    # Example preprocessing steps
    return text.strip().lower()


# Function to validate CSV input
def validate_csv(file):
    """
    Validate that the uploaded file is a valid CSV with the required columns.

    Args:
        file: The uploaded file object.

    Returns:
        DataFrame: A Pandas DataFrame of the CSV data if valid.

    Raises:
        ValueError: If the CSV is invalid or missing the 'text' column.
    """
    try:
        data = pd.read_csv(file)
        if "text" not in data.columns:
            raise ValueError("CSV must contain a 'text' column.")
        return data
    except Exception:
        raise ValueError("CSV must contain a 'text' column.")  # Simplified


# Function to write results to a CSV
def save_results_to_csv(results, output_file):
    """
    Save sentiment analysis results to a CSV file.

    Args:
        results (list of dict): The results from sentiment analysis.
        output_file (str): The path to save the CSV file.

    Returns:
        None
    """
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
