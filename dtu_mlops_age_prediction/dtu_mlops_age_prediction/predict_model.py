import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import age_predictor_model
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

def predict(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns:
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model
    """

    # Set the model to evaluation mode
    model.eval()

    predictions = []

    # Iterate over batches in the dataloader
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch into inputs and labels
            inputs, labels = batch

            # Move data to device if using GPU
            inputs = inputs.to(device) if torch.cuda.is_available() else inputs
            labels = labels.to(device) if torch.cuda.is_available() else labels

            # Forward pass
            outputs = model(inputs)

            # Append predictions
            predictions.append(outputs)

    # Concatenate predictions from all batches
    return torch.cat(predictions, dim=0)

def import_data():
    path_in = 'data/processed'
    test_data = torch.load(Path(path_in + '/test_data.pt'))
    test_labels = torch.load(Path(path_in + '/test_labels.pt'))

    # Convert the list of tensors and labels to PyTorch tensors
    test_data = [torch.tensor(data) for data in test_data]
    
    # Convert labels to numerical format (assuming they are not already)
    test_labels = [label if isinstance(label, (int, float)) else 0 for label in test_labels]
    test_labels = torch.tensor(test_labels, dtype=torch.long)  # Adjust dtype accordingly

    # Create a PyTorch TensorDataset
    dataset = TensorDataset(torch.stack(test_data), test_labels)
    return dataset
# Example usage:
# Assuming you have a model and a dataloader defined

model_path = 'models/model.pt'
model = torch.load(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataloader = DataLoader(import_data())

# Make predictions
predictions = predict(model, dataloader)
print(predictions)
