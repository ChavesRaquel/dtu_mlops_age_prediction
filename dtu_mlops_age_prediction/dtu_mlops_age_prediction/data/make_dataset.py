import os
from pathlib import Path
from torchvision import transforms
from PIL import Image

if __name__ == '__main__':
    # Get the data and process it
    path_in = Path('data/raw/face_age')
    path_out = Path('data/processed')
    labels = []
    data = []
    transformaciones = transforms.Compose([transforms.ToTensor(), ])
    for folder in os.listdir(path_in):
        path = Path(f'data/raw/face_age/{folder}')
        for file in os.listdir(path):
            image = Image.open(f'data/raw/face_age/{folder}/{file}')
            data.append(transformaciones(image))
            
        cont = len(os.listdir(path))
        labels += [folder]*cont
        print(data)
        #print(labels)
        break