from torch import nn


class Age_predictor(nn.Module):
    def __init__(self, num_classes=90):
        super(Age_predictor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 50 * 50, 128),  # Adjust the input size based on your image size
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


age_predictor_model = Age_predictor()
