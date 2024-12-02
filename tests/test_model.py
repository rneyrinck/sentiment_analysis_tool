import pytest

from app.model import load_model, analyze_text, analyze_batch


# Fixture for loading the model once for all tests
@pytest.fixture(scope="module")
def model():
    return load_model()


# Test that the model loads correctly
def test_load_model(model):
    assert model is not None
    assert callable(model)  # Ensure the model is callable


# Test single text analysis
def test_analyze_text_valid(model):
    result = analyze_text(model, "This product is amazing!")
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "confidence" in result
    assert result["sentiment"] in ["positive", "negative", "neutral"]
    assert 0.0 <= result["confidence"] <= 1.0


# Test single text analysis with empty input
def test_analyze_text_empty(model):
    result = analyze_text(model, "")
    assert result["sentiment"] == "neutral"  # Assuming neutral for empty input
    assert result["confidence"] == 0.0


# Test batch text analysis with valid inputs
def test_analyze_batch_valid(model):
    texts = [
        "I love this product!",
        "This is the worst experience ever.",
        "It's fine, nothing special.",
    ]
    results = analyze_batch(model, texts)
    assert isinstance(results, list)
    assert len(results) == len(texts)
    for result in results:
        assert "text" in result
        assert "sentiment" in result
        assert "confidence" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert 0.0 <= result["confidence"] <= 1.0


# Test batch text analysis with empty list
def test_analyze_batch_empty(model):
    results = analyze_batch(model, [])
    assert results == []  # Expect an empty list


# Test batch text analysis with very large input
def test_analyze_batch_large_input(model):
    texts = ["This is great!"] * 1000  # Simulate 1000 inputs
    results = analyze_batch(model, texts)
    assert len(results) == 1000
    for result in results:
        assert "sentiment" in result
        assert "confidence" in result
