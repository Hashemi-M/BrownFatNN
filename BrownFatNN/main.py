from Model.BFNN import *
from DataPipeline.DataCleaning import *
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# set device for training.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))


# Creat NN instance
model = BFNN().to(device)
print(model)

# Set loss function and optimizer 
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(bf_train_dataloader, model, loss_fn, optimizer)
    test(bf_test_dataloader, model)
print("Done!")
