import time
import json
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from vision.trained_models import ChessClassifier
from src.utils.app_utils import AppUtil
import torchvision.models as models
from config.logger import Logger

logger = Logger(__name__)

class ChessCategorizer(nn.Module):
    BASEPATH = Path(__file__).parent.resolve().parent
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    MODEL_PATH = BASEPATH / "vision" / "trained_models"
    MODEL = "vgg16_modelV1.bin"

    """
    A class to categorize chess pieces using a pre-trained VGG16 model and a custom chess classifier model.

    Args:
        model_path (str): Path to the directory containing the model files.
        label_to_chess (dict): A dictionary mapping class labels to chess piece names.
        threshold (float): The threshold for classification probability. Default is 0.3.
    """

    def __init__(
            self,
            device: str = DEVICE,
            model_path = str(MODEL_PATH / MODEL),
            label_to_chess = AppUtil.load_file(MODEL_PATH / "class_labels.json"),
            threshold: float = 0.3
        ):
        super(ChessCategorizer, self).__init__()

        self.in_features = 4096
        self.device = device
        self.num_labels = len(label_to_chess)
        self.label_to_chess = label_to_chess
        self.threshold = threshold

        self.net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.net.classifier[6] = nn.Linear(in_features=self.in_features, out_features=self.num_labels)
        self.net.eval()
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device(device=self.device)))

        logger.info("Loaded dictionary: ", self.label_to_chess)
        logger.info("Using: ", self.device)

    def predict(self, model, input_tensor):
        """
        Predict the class of the input tensor using the given model.

        Args:
            model (nn.Module): The model to use for prediction.
            input_tensor (torch.Tensor): The input tensor to classify.

        Returns:
            tuple: A tuple containing the predicted class, probability, and label.
        """
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        outputs = torch.softmax(output, dim=1)
        preds = torch.argmax(outputs).item()
        probs = outputs[0, preds]
        label = self.label_to_chess[preds]

        return preds, probs, label

    def classify(self, input_pil_image, image_size=224):
        """
        Classify the chess piece in the input PIL image.

        Args:
            input_pil_image (PIL.Image): The input image to classify.

        Returns:
            tuple: A tuple containing the predictions from both models.
        """
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        input_tensor = preprocess(input_pil_image).unsqueeze(0)
        start_time = time.time()

        # Get predictions from both models
        preds, probs, label = self.predict(self.net, input_tensor)

        end_time = time.time()
        latency = end_time - start_time
        throughput = 1 / latency

        logger.info(f"Latency: {latency:.4f} seconds")
        logger.info(f"Throughput: {throughput:.4f} predictions per second")

        return preds, probs, label, latency, throughput