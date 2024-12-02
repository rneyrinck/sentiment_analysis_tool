# Sentiment Analysis API

## Overview
A lightweight and scalable API for analyzing the sentiment of customer feedback or text data. The tool uses a pretrained Hugging Face Transformers model to classify text into **positive**, **negative**, or **neutral** sentiment. It supports single text analysis as well as batch processing through CSV uploads.

---

## Features
- **Single Text Analysis**: Submit a single text string to analyze its sentiment and confidence score.
- **Batch Analysis**: Upload a CSV file containing multiple text inputs, and receive a CSV with sentiment results.
- **Interactive Documentation**: Explore and test the API with Swagger UI.
- **Deploy Anywhere**: Easy deployment on Render.com, Heroku, or any platform supporting Python-based APIs.

---

## Tech Stack
- **Framework**: FastAPI
- **Machine Learning**: Hugging Face Transformers, PyTorch
- **Deployment**: Render.com (primary target), optional Docker support

---

## API Endpoints

### 1. Analyze Single Text
- **Method**: `POST`
- **Endpoint**: `/analyze`
- **Request Body**:
    ```json
    {
        "text": "I love this product!"
    }
    ```
- **Response**:
    ```json
    {
        "sentiment": "positive",
        "confidence": 0.95
    }
    ```

### 2. Analyze Batch (CSV Upload)
- **Method**: `POST`
- **Endpoint**: `/batch`
- **Request**: Upload a CSV file with a `text` column.
- **Response**:
    - A JSON array of analyzed results:
    ```json
    [
        {"text": "I love this!", "sentiment": "positive", "confidence": 0.99},
        {"text": "This is terrible.", "sentiment": "negative", "confidence": 0.98}
    ]
    ```
    - Alternatively, processed CSV output is downloadable.

---

## Setup and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment_analysis_tool.git
cd sentiment_analysis_tool
```
### 2. Set Up a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Start the API Locally
```bash
uvicorn app.main:app --reload
```
The API will be available at http://127.0.0.1:8000.

### 5. Test the API
- Use a tool like Postman or PyCharm's HTTP client to send requests.
- Visit http://127.0.0.1:8000/docs to access the Swagger UI and test the API interactively.

## Deployment
### Deployment on Render.com
1. Push your repository to GitHub, GitLab, or Bitbucket.
2. Create a new Web Service on Render.com.
3. Configure the service:
- Build Command: ```bash pip install -r requirements.txt```
- Start Command: ```bash uvicorn app.main:app --host 0.0.0.0 --port $PORT```
4. Deploy and access your application at the provided URL.
### Optional: Docker Setup
1. Build the Docker image:
```bash
docker build -t sentiment-analysis-api
```
2. Run the Docker container:
```bash
docker run -p 8000:8000 sentiment-analysis-api
```
Access the API at http://127.0.0.1:8000.
## Project Structure
```plaintext
sentiment-analysis-api/
│
├── README.md                # Overview and instructions
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files and folders
│
├── app/                     # Main application code
│   ├── __init__.py          # Initialization file
│   ├── main.py              # FastAPI entry point
│   ├── model.py             # Sentiment analysis functions
│   └── utils.py             # CSV validation and preprocessing
│
├── tests/                   # Unit and integration tests
│   ├── test_api.py          # API endpoint tests
│   └── test_model.py        # Sentiment analysis model tests
│
├── data/                    # Sample files
│   ├── sample_reviews.csv   # Example input CSV for batch processing
│   └── processed_output.csv # Example processed output
│
└── Dockerfile               # Optional Docker configuration
```
## Contributing
Contributions are welcome! Feel free to:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.