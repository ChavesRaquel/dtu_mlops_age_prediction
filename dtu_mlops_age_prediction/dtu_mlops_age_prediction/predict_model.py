import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import age_predictor_model
from pathlib import Path

def predict(
    model: nn.Module,
    dataloader: DataLoader
) -> torch.Tensor:
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
            # Move data to device if using GPU
            inputs = batch.to(device) if torch.cuda.is_available() else batch

            # Forward pass
            outputs = model(inputs)

            # Append predictions
            predictions.append(outputs.cpu())  # Move predictions back to CPU

    # Concatenate predictions from all batches
    return torch.cat(predictions, dim=0)

def import_data():
    path_in = 'data/processed'
    test_data = torch.load(Path(path_in +'/test_data.pt'))
    #test_data = np.array(test_data)
    test_label = torch.load(Path(path_in + '/test_labels.pt'))
    #test_label =  np.array(test_label)
    label_to_index = {label: idx for idx, label in enumerate(set(test_label))}
    test_label = [label_to_index[label] for label in test_label]

    # Convert the list of tensors and numerical labels to PyTorch tensors
    test_data = [torch.tensor(data) for data in test_data]
    test_label = torch.tensor(test_label)
    

    # Create a PyTorch TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.stack(test_data), test_label)
    return dataset

# Example usage:
# Assuming you have a model and a dataloader defined

model = age_predictor_model # Load the pretrained model state_dictmodel_path = 'path/to/your/model.pt'model.load_state_dict(torch.load(model_path)) # Set the model to evaluation modemodel.eval()

model = torch.load('models/model.pt')
dataloader = import_data()

# Make predictions
predictions = predict(model, dataloader)
print(predictions)