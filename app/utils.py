import pandas as pd


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
    except Exception as e:
        raise ValueError(f"Invalid CSV file: {str(e)}")
