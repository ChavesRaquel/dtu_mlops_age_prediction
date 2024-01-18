#import click
import torch
from pathlib import Path
from torch import nn
from models.model import age_predictor_model
#import numpy as np
import hydra
import wandb
import os
import yaml

#from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def import_data(train = True):
    
    path_in = hydra.utils.get_original_cwd()+'/data/processed'
    if train is True:
        data = torch.load(Path(path_in +'/train_data.pt'))
        label = torch.load(Path(path_in + '/train_labels.pt'))
    else:
        data = torch.load(Path(path_in +'/test_data.pt'))
        label = torch.load(Path(path_in + '/test_labels.pt'))

    label_to_index = {lab: idx for idx, lab in enumerate(set(label))}
    label = [label_to_index[lab] for lab in label]

    # Convert the list of tensors and numerical labels to PyTorch tensors
    data = [torch.tensor(dat) for dat in data]
    label = torch.tensor(label)
    

    # Create a PyTorch TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.stack(data), label)
    return dataset

def calculate_validation_accuracy(model, val_dataloader):
    model = age_predictor_model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for val_batch in val_dataloader:
            val_x, val_y = val_batch
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_y_pred = model(val_x)
            _, predicted = torch.max(val_y_pred.data, 1)
            total += val_y.size(0)
            correct += (predicted == val_y).sum().item()

    accuracy = correct / total

    return accuracy

@hydra.main(config_name="config_train.yaml", config_path='../config')
def train(cfg):
    print("Current Working Directory:", hydra.utils.get_original_cwd())
    learning_rate = cfg.hyperparameters.learning_rate
    batch_size = cfg.hyperparameters.batch_size
    num_epochs = cfg.hyperparameters.num_epochs

    wandb.init(project='Logging')
    model = age_predictor_model.to(device)
    model.train()
    
    wandb.watch(model, log_freq =100)

    #Importing train set
    train_set = import_data(train=True)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    #Importing test set
    val_set = import_data(train=False)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
     
        
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            train_loss = loss_fn(y_pred, y)
            train_loss.backward()
            optimizer.step()
        #Log training loss
        wandb.log({'train_loss': train_loss})

        #Calculate and log training accuracy
        train_acc = calculate_validation_accuracy(model, train_dataloader)
        wandb.log({'train_acc': train_acc})
        #Calculate and log validation accuracy
        val_acc = calculate_validation_accuracy(model, val_dataloader)
        wandb.log({'val_acc': val_acc})

        #Log epoch
        

        print(f"Epoch {epoch} Train Loss {train_loss} Validation Accuracy {val_acc}")

    torch.save(model, hydra.utils.get_original_cwd()+"/models/model.pt")

if __name__ == "__main__":
    train()

