import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np


# Load the model
model_path: str = 'dtu_mlops_age_prediction/models/model.pt'
model: torch.nn.Module = torch.load(model_path)
model.eval()

# Load testing data
data_pat: str = 'data/processed/test_data.pt'
labels_path: str = 'data/processed/test_labels.pt'

test_data: List[torch.Tensor] = torch.load(data_path)
test_labels: List[str] = torch.load(labels_path)

# Assuming test_data is a list of image tensors
# Assuming test_labels is a list of label strings

# Define the transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Create a DataLoader for the testing data
batch_size: int = 32  # Adjust as needed
test_dataset: TestDataset = [(transform(img), label) for img, label in zip(test_data, test_labels)]
test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Test the model
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
correct_predictions: int = 0
diferencias: List[List[int]] = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)


        predicted = predicted.tolist()
        labels = [int(x) if x.isdigit() else 90 for x in labels]
 #       print(predicted)
#        print(labels)
        correct_predictions +=  sum(item1 == item2 for item1, item2 in zip(predicted, labels))    
        diferencias.append([abs(elemento1 - elemento2) for elemento1, elemento2 in zip(predicted, labels)])


accuracy: float = correct_predictions / len(test_loader.dataset)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
print(f"Mean error: {np.mean([elemento for sublista in diferencias for elemento in sublista]):.2f}")
