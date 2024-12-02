# Sentiment Analysis API

## Overview
A lightweight and scalable API for analyzing customer feedback sentiment. The tool classifies text into **positive**, **negative**, or **neutral** sentiment and can process single inputs or batch uploads.

---

## Features
- **Single Input Analysis**: Accepts a text string and returns the sentiment with a confidence score.
- **Batch Processing**: Processes a CSV file of multiple text inputs and returns sentiment analysis results.
- **REST API**: Easy integration with existing workflows.
- **Deployable Anywhere**: Dockerized and ready for deployment on Heroku, AWS, or GCP.

---

## Tech Stack
- **Backend**: Python, FastAPI
- **Machine Learning**: Hugging Face Transformers, Scikit-learn
- **Deployment**: Docker, Heroku

---

## API Endpoints

### 1. Analyze Single Text
- **Endpoint**: `POST /analyze`
- **Input**:
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
- **Endpoint**: `POST /batch`
- **Input**: Upload a CSV with a `text` column.
- **Output**: Downloadable CSV with an added `sentiment` column.

---

sentiment-analysis-api/
│
├── README.md                # Overview and instructions
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files and folders
│
├── app/                     # Main application code
│   ├── __init__.py          # Initialization file for the app
│   ├── main.py              # API entry point (FastAPI/Flask app)
│   ├── model.py             # Sentiment analysis pipeline
│   └── utils.py             # Helper functions (e.g., data preprocessing)
│
├── tests/                   # Unit and integration tests
│   ├── test_api.py          # Tests for API endpoints
│   └── test_model.py        # Tests for ML pipeline
│
├── data/                    # Sample input and output files
│   ├── sample_reviews.csv   # Example CSV for batch processing
│   └── processed_output.csv # Example output for documentation/demo
│
├── docker/                  # Docker-related configuration
│   ├── Dockerfile           # Dockerfile to containerize the app
│   └── docker-compose.yml   # Optional: Multi-container setup
│
└── deployment/              # Deployment scripts/configs
    ├── heroku.yml           # Heroku-specific config (if applicable)
    └── README.md            # Deployment instructions


## Setup and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/RNEYRINCK/sentiment_analysis_api.git
cd sentiment-analysis-api
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Run the App
```bash
uvicorn app.main:app --reload
The API will be available at http://127.0.0.1:8000.
```
## Deployment
### Using Docker
1. Build the Docker image:
```bash
docker build -t sentiment-analysis-api .
```
2. Run the container:
```bash
docker run -p 8000:8000 sentiment-analysis-api
```
### Using Heroku
1. Install Heroku CLI.
2. Deploy the app:
```bash
heroku create
git push heroku main
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.