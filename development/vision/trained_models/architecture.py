import torch
import torch.nn as nn

class ChessClassifier(nn.Module):
    """
    A Convolutional Neural Network (CNN) for classifying chess pieces.

    The architecture consists of three convolutional layers followed by three fully connected (dense) layers.
    Each convolutional layer includes a convolution operation, max pooling, batch normalization, ReLU activation,
    and dropout. The fully connected layers include linear transformations, ReLU activation, and dropout (except the final layer).

    Attributes:
        conv1 (nn.Sequential): First convolutional layer.
        conv2 (nn.Sequential): Second convolutional layer.
        conv3 (nn.Sequential): Third convolutional layer.
        fc1 (nn.Sequential): First fully connected layer.
        fc2 (nn.Sequential): Second fully connected layer.
        fc3 (nn.Sequential): Third fully connected layer (output layer).
    """

    def __init__(self, num_labels):
        """
        Initializes the ChessClassifier with the specified number of output labels.

        Args:
            num_labels (int): The number of output labels (classes).
        """
        super(ChessClassifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # (224 - 4 + 2 * 1) / 2 + 1 = 112
            nn.MaxPool2d(kernel_size=4, stride=4), # (112 - 4) / 4 + 1 = 28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1), # (28 - 2 + 2 * 1) / 2 + 1 = 15
            nn.MaxPool2d(kernel_size=3, stride=3), # (15 - 3) / 3 + 1 = 5
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1), # (5 - 2 + 2 * 1) / 1 + 1 = 6
            nn.MaxPool2d(kernel_size=6, stride=1), # (6 - 6) / 1 + 1 = 1
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x