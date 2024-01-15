import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


# Load the model
model_path = 'models/model.pt'
model = torch.load(model_path)
model.eval()

# Load testing data
data_path = 'data/processed/train_data.pt'
labels_path = 'data/processed/train_labels.pt'

test_data = torch.load(data_path)
test_labels = torch.load(labels_path)

# Assuming test_data is a list of image tensors
# Assuming test_labels is a list of label strings

# Define the transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Create a DataLoader for the testing data
batch_size = 32  # Adjust as needed
test_dataset = [(transform(img), label) for img, label in zip(test_data, test_labels)]
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Test the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
correct_predictions = 0
diferencias = []

with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("/log/pred_age_model")) as prof:
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)


            predicted = predicted.tolist()
            labels = [int(x) if x.isdigit() else 90 for x in labels]
            
            correct_predictions +=  sum(item1 == item2 for item1, item2 in zip(predicted, labels))    
            diferencias.append([abs(elemento1 - elemento2) for elemento1, elemento2 in zip(predicted, labels)])

prof.export_chrome_trace("/predict_trace.json")
accuracy = correct_predictions / len(test_loader.dataset)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
print(f"Mean error: {np.mean([elemento for sublista in diferencias for elemento in sublista]):.2f}")
