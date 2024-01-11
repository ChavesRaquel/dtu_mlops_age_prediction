from torch import nn

age_predictor_model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Flatten(),
    
    nn.Linear(64 * 16 * 16, 128),
    nn.ReLU(),
    
    nn.Linear(128, 90),  # 90 classes for age prediction
    nn.LogSoftmax(dim=1)    
)