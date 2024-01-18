import timm
import torch.nn.functional as F
import torch.nn as nn

class Age_predictor_timm(nn.Module):
    def __init__(self, num_classes=90, model_name='resnet18', pretrained=True):
        super(Age_predictor_timm, self).__init__()

        # Load the pre-trained backbone from timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Modify classifier based on the number of output classes
        in_features = self.backbone.num_features
        self.backbone.reset_classifier(num_classes=0)  # Remove the original classifier

        # New classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Feature extraction using the backbone
        x = self.backbone(x)

        x = x.unsqueeze(2).unsqueeze(3)
        
        # Global average pooling (GAP) can be used instead of Flatten
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Squeeze singleton dimensions if they exist
        x = x.squeeze(-1).squeeze(-1)
        
        # Classification using the new classifier
        x = self.classifier(x)
        return x


# Example usage with ResNet18
age_predictor_model = Age_predictor_timm(model_name='resnet18', num_classes=90, pretrained=True)
