from typing import List, Tuple
import os
from pathlib import Path
from torchvision import transforms
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

def process_data() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    path_in: Path = Path('data/raw/face_age/')
    path_out: str = 'data/processed'
    data: List[torch.Tensor] = []
    labels: List[str] = []
    train_data: List[torch.Tensor] = []
    test_data: List[torch.Tensor] = []
    train_labels: List[str] = []
    test_labels: List[str] = []
    transformations = transforms.Compose([transforms.ToTensor(), ])
    
    # Go through image folders
    for folder in os.listdir(path_in):
        path: Path = Path(f'data/raw/face_age/{folder}')

        # Skip non-important folders
        if folder == 'face_age':
            continue

        # Open images in each folder
        for file in os.listdir(path):
            image: Image.Image = Image.open(f'data/raw/face_age/{folder}/{file}')
            data.append(transformations(image))

        # Set age 90+ because there is too little data to go year by year
        if int(folder) >= 90:
            folder = '90+'
        
        # Define the labels
        cont: int = len(os.listdir(path))
        labels += [folder] * cont
    
    # Divide in training and test
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)

    return train_data, test_data, train_labels, test_labels

if __name__ == '__main__':
    train_data, test_data, train_labels, test_labels = process_data()

    # Save the data    
    torch.save(train_data, Path('data/processed/train_data.pt'))
    torch.save(test_data, Path('data/processed/test_data.pt'))
    torch.save(train_labels,  Path('data/processed/train_labels.pt'))
    torch.save(test_labels, Path('data/processed/test_labels.pt'))
