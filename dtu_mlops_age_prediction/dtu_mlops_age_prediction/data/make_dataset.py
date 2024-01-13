import os
from pathlib import Path
from torchvision import transforms
import torch
from PIL import Image
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Get the data and process it
    path_in = Path("data/raw/face_age/")
    path_out = "data/processed"
    data = []
    labels = []
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Go through image folders
    for folder in os.listdir(path_in):
        path = Path(f"data/raw/face_age/{folder}")

        # Skip non important folders
        if folder == "face_age":
            continue

        # Open images in each folder
        for file in os.listdir(path):
            image = Image.open(f"data/raw/face_age/{folder}/{file}")
            data.append(transformations(image))

        # Set age 90+ because there is too little data to go year by year
        if int(folder) >= 90:
            folder = "90+"

        # Define the labels
        cont = len(os.listdir(path))
        labels += [folder] * cont

    # Divide in training and test
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)

    # Save the data
    torch.save(train_data, Path(path_out + "/train_data.pt"))
    torch.save(test_data, Path(path_out + "/test_data.pt"))
    torch.save(train_labels, Path(path_out + "/train_labels.pt"))
    torch.save(test_labels, Path(path_out + "/test_labels.pt"))
