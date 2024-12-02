from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
import logging

logging.basicConfig(level=logging.DEBUG)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sentiment Analysis API is running!"}


def test_analyze_sentiment_valid():
    response = client.post("/analyze", json={"text": "I absolutely love this product!"})
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert "confidence" in response.json()
    assert response.json()["sentiment"] in ["positive", "negative", "neutral"]


def test_analyze_sentiment_empty():
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422  # Unprocessable Entity for empty text


def test_batch_sentiment_valid():
    file_data = "text\nI love this!\nThis is terrible.\nIt's okay.\n"
    response = client.post(
        "/batch",
        files={"file": ("test.csv", file_data, "text/csv")},
    )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 3
    for result in results:
        assert "text" in result
        assert "sentiment" in result
        assert "confidence" in result


def test_batch_sentiment_invalid_file():
    file_data = "invalid data\nwithout a text column\n"
    response = client.post(
        "/batch",
        files={"file": ("invalid.csv", file_data, "text/csv")},
    )
    assert response.status_code == 200  # API should handle gracefully
    assert response.json()["error"] == "CSV must contain a 'text' column."
