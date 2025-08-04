import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PolygonDataset
from unet import UNet
import wandb
import os

wandb.init(project="colored-polygon-unet")

train_dir = "dataset/training"
val_dir = "dataset/validation"

train_dataset = PolygonDataset(
    input_dir=os.path.join(train_dir, "inputs"),
    output_dir=os.path.join(train_dir, "outputs"),
    json_path=os.path.join(train_dir, "data.json")
)

val_dataset = PolygonDataset(
    input_dir=os.path.join(val_dir, "inputs"),
    output_dir=os.path.join(val_dir, "outputs"),
    json_path=os.path.join(val_dir, "data.json")
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = UNet().cuda()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 25
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x, y, mask in train_loader:
        x, y, mask = x.cuda(), y.cuda(), mask.cuda()
        optimizer.zero_grad()
        output = model(x)
        loss = (criterion(output, y) * mask).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    wandb.log({"train_loss": train_loss / len(train_loader), "epoch": epoch})

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            output = model(x)
            loss = (criterion(output, y) * mask).mean()
            val_loss += loss.item()
    wandb.log({"val_loss": val_loss / len(val_loader), "epoch": epoch})

torch.save(model.state_dict(), "unet_polygon.pth")
