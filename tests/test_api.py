from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test the root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sentiment Analysis API is running!"}

# Test the /analyze endpoint with valid input
def test_analyze_sentiment_valid():
    response = client.post("/analyze", json={"text": "I absolutely love this product!"})
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert "confidence" in response.json()
    assert response.json()["sentiment"] in ["positive", "negative", "neutral"]

# Test the /analyze endpoint with empty input
def test_analyze_sentiment_empty():
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422  # Unprocessable Entity (invalid input)

# Test the /batch endpoint with a valid CSV
def test_batch_sentiment_valid():
    # Create a temporary CSV file for testing
    file_data = "text\nI love this!\nThis is terrible.\nIt's okay.\n"
    response = client.post(
        "/batch",
        files={"file": ("test.csv", file_data, "text/csv")},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for result in response.json():
        assert "text" in result
        assert "sentiment" in result
        assert "confidence" in result

# Test the /batch endpoint with invalid file
def test_batch_sentiment_invalid_file():
    file_data = "invalid data\nwithout a text column\n"
    response = client.post(
        "/batch",
        files={"file": ("invalid.csv", file_data, "text/csv")},
    )
    assert response.status_code == 400
    assert "error" in response.json()
