import sys

import torch
import pytest
from src.models.model import Age_predictor_timm

@pytest.fixture
def model():
    # Create an instance of model for testing
    return Age_predictor_timm(model_name='resnet18', num_classes=90, pretrained=True)

def test_model_architecture(model):
    # Test the architecture of the model
    assert isinstance(model.backbone, torch.nn.Module), "Backbone should be a torch.nn.Module"
    assert isinstance(model.classifier, torch.nn.Sequential), "Classifier should be a Sequential module"

def test_model_output(model):
    # Test the forward pass of the model with dummy input
    dummy_input = torch.randn(1, 3, 200, 200)  # Batch size of 1, 3 channels, 200x200 image
    output = model(dummy_input)

    # Check if the output has the expected shape
    assert output.shape == torch.Size([1, 90]), "Incorrect output shape"

