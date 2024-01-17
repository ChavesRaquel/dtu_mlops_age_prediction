import click
import torch
from pathlib import Path
from torch import nn
from models.model import age_predictor_model
import numpy as np
import wandb
import os
import yaml

from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def import_data():
    path_in = os.getcwd()+'/data/processed'
    train_data = torch.load(Path(path_in +'/train_data.pt'))
    train_label = torch.load(Path(path_in + '/train_labels.pt'))

    label_to_index = {label: idx for idx, label in enumerate(set(train_label))}
    train_label = [label_to_index[label] for label in train_label]

    # Convert the list of tensors and numerical labels to PyTorch tensors
    train_data = [torch.tensor(data) for data in train_data]
    train_label = torch.tensor(train_label)
    

    # Create a PyTorch TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.stack(train_data), train_label)
    return dataset


def train(learning_rate, batch_size, num_epochs):
    
    model = age_predictor_model.to(device)
    wandb.watch(model, log_freq =100)

    train_set = import_data()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
     
        
    for epoch in range(num_epochs):
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
        wandb.log({'loss':loss})

    torch.save(model, "/models/model.pt")

def main():
    project_path=os.getcwd()
    print("Current working directory: ", project_path)
    print("Training with sweep")

    with open(project_path + '/config/wandb_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    wandb.init(config=config,project="Sweep1")
    sweep_id = wandb.sweep(sweep=config, project="Sweep1")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Start Training...")

    # Extract values from config file
    lr = wandb.config.lr
    batch_size = wandb.config.batch_size
    n_epoch = wandb.config.epochs
    
    train(learning_rate=lr, batch_size=batch_size, num_epochs=n_epoch)

if __name__ == "__main__":
    main()

