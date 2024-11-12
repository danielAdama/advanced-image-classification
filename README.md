# Chess Classification API

This repository hosts a custom-trained PyTorch model API for classifying chess images. The application uses FastAPI for backend services, NGINX for frontend delivery, and Docker for containerization.

## Project Structure

The project has the following structure:

```
development/
├── config/                   # Configuration files
│   ├── app_config.yml        # App configurations
│   ├── config_helper.py      # Config helper functions
│   ├── nginx.conf            # NGINX server configuration
│   └── logger.py             # Logging setup
├── frontend/                 # Frontend UI
│   ├── index.html            # HTML page for the web interface
│   ├── script.js             # JavaScript for API interactions
│   └── styles.css            # Styling for the frontend
├── notebook/                 # Jupyter notebooks
│   └── chess_classify.ipynb  # Model training and evaluation notebook
├── src/                      # Source code for the API
│   ├── main.py               # FastAPI application entry
│   ├── controllers/          # API route controllers
│   ├── services/             # Core service logic
│   ├── vision/               # Model and utilities for image classification
│   └── ...
├── Dockerfile.backend        # Dockerfile for backend
├── Dockerfile.frontend       # Dockerfile for frontend
├── docker-compose.yml        # Docker Compose file
├── LICENSE                   # License file
├── Makefile                  # Task management
└── README.md                 # Project documentation (this file)
```

## Setup

### Prerequisites

- Docker and Docker Compose installed.
- Python 3.11+ (for local development).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danielAdama/advanced-image-classification.git
   cd chess-classification
   ```

2. **Set up Docker containers**  
   Build and start the containers for the frontend and backend.
   ```bash
   docker-compose up -d
   ```

3. **Verify**  
   - The backend API should be running at `http://localhost:8001/docs`.
   - The frontend should be accessible at `http://localhost:8080`.

### Configuration

- `config/app_config.yml`: Main app configuration settings.
- `config/nginx.conf`: NGINX configuration for proxying frontend and backend traffic.

## API Endpoints

- **POST** `/v1/image/classify/`  
  Upload an image for classification.

## Usage

The API allows users to upload images and get a classification with details on prediction, probability, latency, and throughput. Users can interact through the frontend UI or via API calls.

### Frontend Interaction

Upload an image on the frontend UI, and it will display:
- Prediction and probability of the classified label.
- Model latency and throughput metrics.

### Makefile Commands

- **Run Docker Compose**  
  ```bash
  make compose_up
  ```
- **Shut down Docker Compose**  
  ```bash
  make compose_down
  ```

## Model

The model is built and trained using `torch` and saved in the `src/vision/trained_models` folder. Details on training are in the `notebook/chess_classify.ipynb` file.

## License

This project is licensed under the MIT License.
