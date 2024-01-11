import click
import torch
from pathlib import Path
from torch import nn
from models.model import myawesomemodel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=64, help="batch size to use for training")
@click.option("--num_epochs", default=5, help="number of epochs to train for")
def train(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(batch_size)

def import_data():
    path_in = 'data/processed'
    train_data = torch.load(Path(path_in +'/train_data.pt'))
    train_label = torch.load(Path(path_in + '/train_labels.pt'))

    return torch.utils.data.TensorDataset(train_data, train_label)


    
model = myawesomemodel.to(device)
train_set, _ = import_data()
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

torch.save(model, "model.pt")



cli.add_command(train)



if __name__ == "__main__":
    cli()
