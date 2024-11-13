# Chess Classification API

Front End Interface
<img width="1680" alt="Screenshot 2024-11-12 at 23 44 53" src="https://github.com/user-attachments/assets/ba7d0bcd-2eda-4791-8f74-a0852a1a186f">

<img width="1680" alt="Screenshot 2024-11-12 at 23 44 25" src="https://github.com/user-attachments/assets/633219f2-c0c9-494d-881e-21f7bc0a7062">

<img width="1680" alt="Screenshot 2024-11-12 at 23 44 05" src="https://github.com/user-attachments/assets/949779cc-2f33-4a99-9d32-572f956c7b3e">

Backend Interface
<img width="1680" alt="Screenshot 2024-11-12 at 23 45 29" src="https://github.com/user-attachments/assets/7d0b2039-8685-466e-97c8-7073eaf85eed">


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
Technology Stack
Backend:

FastAPI: Backend framework for building APIs.
Docker: Containerization for easy deployment.
Pytorch: Deep Learning Model Development

Frontend: HTML, CSS, JavaScript: Simple web-based frontend for chat interaction.

## Setup

### Prerequisites

- Docker and Docker Compose installed.
- Python 3.11+ (for local development).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danielAdama/advanced-image-classification.git
   cd chess-classification/development
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

1.0 Data Preparation 

Unzipping the Training Data 

I started by unzipping the dataset and listing the classes. The dataset consists of images categorized into the following classes:   

- Knight 

- Bishop 

- King 

- Queen 

- Rook 

- Pawn 

These represent the chess pieces we want the model to recognize. 


1.1 Data Augmentation 

Data augmentation involves applying transformations to the training images to generate additional samples. This helps to diversify the dataset and improve the model’s ability to generalize (i.e., perform well on new, unseen data).   

For this task, I used an ‘ImageAugmentation’ class to apply various transformations to the dataset. 


 1.2 Dataset Splitting 


Dataset Class 

The ‘ImageDataset’ class is responsible for loading and transforming images during training.   


Dataset Splitting 

The dataset was divided into three parts:   

- Training set: 94% of the data   

- Validation set: 6% of the data   

- Test set: 30 samples (used for final model evaluation) 


This ensures that the model has enough data to learn from, while also having data it hasn’t seen during training to test its performance. 

 2.0 Model Selection and Enhancement 

Baseline Model 

The ‘ChessClassifier’ class defines a Convolutional Neural Network (CNN) architecture with convolutional and fully connected layers. The CNN architecture is a standard choice for image classification tasks. 

  

Training the Baseline Model 

The baseline model was trained for 25 epochs. However, due to the complexity of the dataset and the simplicity of the architecture, the model didn't improve significantly. The final performance was only 30.8% accuracy. This indicates that the simple model struggled to capture the intricacies of the chess pieces' images. 

![metric1](https://github.com/user-attachments/assets/ae628efe-9cc0-4768-8055-fa22bd9a9daf)
Fig 2.1: Training Metrics 

![predm1](https://github.com/user-attachments/assets/3e21e58a-91c9-4d4d-abf3-8cedc1253f56)
 Fig 2.2: Baseline Model Predictions 

 
3.0 Fine-Tuning Pre-Trained Model - VGG16  

Model Setup 

The `setup_vgg16_for_finetuning` function sets up the pre-trained VGG16 model to be fine-tuned for the chess piece classification task. Fine-tuning is an excellent approach when the dataset is small, as it allows the model to leverage knowledge learned from a much larger dataset (like ImageNet) to improve its performance on our task. 
 

Training and Evaluation 

The VGG16 model was trained for 20 epochs, and both training and validation losses and accuracies were recorded and plotted. This helped in visualizing the model’s performance over time.

![metric2](https://github.com/user-attachments/assets/c1815479-6f18-4570-8ca4-15eb6820f2b1)
 Fig 3.1: Training Metrics 

![predm2](https://github.com/user-attachments/assets/f9773336-80cb-47ba-85d0-29b64178675b)
Fig 3.2: Baseline Model Predictions 


Results 

The VGG16 model significantly outperformed the baseline model, achieving a final validation accuracy of 96.2%. This indicates that the fine-tuned model can generalize well to unseen data and is much more effective in recognizing chess pieces. 

  

4.0 Analysis and Improvements 

Data Augmentation 

While the data augmentation process was applied, additional transformations (e.g., scaling) could further diversify the dataset and improve the model’s robustness. It’s important to ensure that these augmented images still resemble realistic chess pieces. 

Model Architecture 

Future improvements could involve experimenting with deeper or more complex network architectures like ResNet. Transfer learning (using pre-trained models) also proved successful with VGG16, and similar approaches could be tested with other models. 

Training Enhancements 

To avoid overfitting (where the model performs well on training data but poorly on new data), it’s recommended to implement early stopping. This technique stops training once the model's performance on the validation set stops improving. 

Evaluation Metrics 

In addition to accuracy, it’s important to use other evaluation metrics like precision, recall, and F1-score. These metrics provide a more detailed understanding of how well the model is performing, especially if there’s an imbalance in the classes. Visualizing the confusion matrix would help in identifying which classes the model struggles with. 

Hyperparameter Tuning 

Experimenting with different learning rates, batch sizes, and other hyperparameters could optimize the model’s performance further.  

5.0 Conclusion 

The notebook is well-structured and successfully demonstrates the steps required to build a chess piece classifier. The VGG16 model performed the best, achieving a final validation accuracy of 96.2%. By implementing the suggested improvements and fine-tuning further, the model's performance can be enhanced even more, ensuring it provides reliable and accurate predictions. 

## License

This project is licensed under the MIT License.
