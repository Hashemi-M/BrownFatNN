import pandas as pd
import os
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# Get Data Path name

fname = os.path.dirname(__file__) + '\..\Data\BrownFat.xls'

# Read excel to pandas

bf_data = pd.read_excel(fname)

# list of significant variables from STAC51 report
sig_col = ["Ext_Temp","2D_Temp","3D_Temp","Weigth","BMI","Size","BrownFat"]

# Extract significant
bf_sig = bf_data[sig_col]
print(bf_sig.shape)

# Drop nan values
bf_sig.dropna()
print(bf_sig.shape)

# Train and test split
train_dat, test_dat = train_test_split(bf_sig, test_size=0.2)

# Split train data matrix into features and label
train_x = train_dat.iloc[:, 0:(train_dat.shape[1]-1)]
train_y = train_dat.iloc[:,train_dat.shape[1]-1]

print(train_x.shape)
print(train_y.shape)

# Split test data matrix into features and label
test_x = test_dat.iloc[:, 0:(test_dat.shape[1]-1)]
test_y = test_dat.iloc[:,test_dat.shape[1]-1]

print(test_x.shape)
print(test_y.shape)

### Converting to data to tensor DataSets and Dataloaders

# Convert np arrays to tensors
train_x_tensor = torch.from_numpy(train_x.values)
train_y_tensor = torch.from_numpy(train_y.values)

# Create DataSets object
bf_train_dataset = TensorDataset(train_x_tensor,train_y_tensor.unsqueeze(1))
# Create DataLoader Object
bf_train_dataloader = DataLoader(bf_train_dataset, batch_size = 64)

print(train_x_tensor)

# Convert np arrays to tensors for test
test_x_tensor = torch.from_numpy(test_x.values)
test_y_tensor = torch.from_numpy(test_y.values)

# Create DataSets object
bf_test_dataset = TensorDataset(test_x_tensor,test_y_tensor.unsqueeze(1))
# Create DataLoader Object
bf_test_dataloader = DataLoader(bf_test_dataset, batch_size = 64)
