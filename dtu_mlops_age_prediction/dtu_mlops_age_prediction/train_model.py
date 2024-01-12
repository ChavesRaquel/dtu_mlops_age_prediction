import click
import torch
from pathlib import Path
from torch import nn
from models.model import age_predictor_model
import numpy as np
import hydra

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def import_data():
    path_in = hydra.utils.get_original_cwd()+'/data/processed'
    train_data = torch.load(Path(path_in +'/train_data.pt'))
    #train_data = np.array(train_data)
    train_label = torch.load(Path(path_in + '/train_labels.pt'))
    #train_label =  np.array(train_label)
    label_to_index = {label: idx for idx, label in enumerate(set(train_label))}
    train_label = [label_to_index[label] for label in train_label]

    # Convert the list of tensors and numerical labels to PyTorch tensors
    train_data = [torch.tensor(data) for data in train_data]
    train_label = torch.tensor(train_label)
    

    # Create a PyTorch TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.stack(train_data), train_label)
    return dataset

@hydra.main(config_name="config_train.yaml", config_path='.')
def train(cfg):
    print("Current Working Directory:", hydra.utils.get_original_cwd())
    print("Config:", cfg)
    model = age_predictor_model.to(device)
    train_set = import_data()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.hyperparameters.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.hyperparameters.num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss {loss}")

    torch.save(model, hydra.utils.get_original_cwd()+"/models/model.pt")



if __name__ == "__main__":
    train()

