"""
Data cleaning of US altitude data from Kaggle
to be used in CNN
https://www.kaggle.com/code/meenakshiramaswamy/geolifeclef2022-basemodel and Zachary Smith
"""
import gc
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from pytorch_dataset import GeoLifeCLEF2022Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


DATA_PATH = Path("/Users/maddiewoolley/Desktop/GEO")
le = LabelEncoder()

dataset = GeoLifeCLEF2022Dataset(DATA_PATH, subset="train",
                                 region='us',
                                 patch_data='landcover',
                                 use_rasters=False,
                                 # transform=get_train_transforms(),
                                 transform=None,
                                 patch_extractor=None)
N_classes = 17036
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)

# Simple Model :)

from torchvision.models.resnet import ResNet, BasicBlock


class ResNetGeolife(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=N_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)


net = ResNetGeolife().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = net.to(device)


def loss_fn(preds, labels):
    loss = nn.CrossEntropyLoss()(preds, labels)
    # loss = nn.BCEWithLogitsLoss()
    return loss


def train(model, optimizer, train_loader, val_loader, epochs=2, device='cpu'):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.float()
            targets = targets.to(device)

            inputs = torch.nan_to_num(inputs)
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            # for near_ir img
            inputs = np.repeat(inputs[..., np.newaxis], 3, -1)
            if inputs.size(1) > 3:
                inputs = inputs.permute(0, 3, 2, 1)
                # print(inputs.shape, inputs.size(1))
                inputs = inputs.to(device)

            output = model(inputs)
            # print(output.shape)
            # print(targets)
            loss = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # training_loss += loss.data.item()*inputs.size(0)
            training_loss += loss.item()
        training_loss /= len(train_loader.dataset)
        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.float()

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            inputs = np.repeat(inputs[..., np.newaxis], 3, -1)
            if inputs.size(1) > 3:
                inputs = inputs.permute(0, 3, 2, 1)
                inputs = inputs.to(device)
                # inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)
                # valid_loss += loss.data.item()*inputs.size(0)
                valid_loss += loss.item()
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                # print (correct)
                num_correct += torch.sum(correct).item()
                # print (num_correct)
                num_examples += correct.shape[0]
                # print (num_examples)
            valid_loss /= len(val_loader.dataset)

            try:
                x = num_correct / num_examples
            except ZeroDivisionError:
                x = 0
            print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, '
                  'accuracy = {:.2f}'.format(epoch + 1, training_loss, valid_loss, x))


print(torch.cuda.get_device_name(0))
print(device)
gc.collect()
train(model.to(device), optimizer, train_loader, val_loader, epochs=2, device=device)
