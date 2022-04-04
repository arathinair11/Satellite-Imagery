import pandas as pd
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import os
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from torchvision.utils import save_image
import torchvision
import random
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import numpy
import seaborn as sns
import tqdm
from sklearn.metrics import classification_report, confusion_matrix

num_epochs =100
batch_size = 32
learning_rate = 1e-3


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading labeled image dataset
root = '/Users/arathi/Desktop/Pycharm/Autoencoder/Labeled_Images/Train'
test = '/Users/arathi/Desktop/Pycharm/Autoencoder/Labeled_Images/Test'
transform = transforms.Compose([transforms.ToTensor()])

def convert(filename):
    return Image.open(filename).convert('RGB')

model_dataset = torchvision.datasets.ImageFolder(root, transform=transform, loader = convert)
test_dataset = torchvision.datasets.ImageFolder(test, transform=transform, loader = convert)

train_count = int(0.7 * len(model_dataset))
val_count = len(model_dataset) - train_count

indices = list(range(len(model_dataset)))
random.seed(123)  # fix the seed so the shuffle will be the same everytime
random.shuffle(indices)

def get_subset(indices, start, end):
    return indices[start : start + end]

train_indices = get_subset(indices, 0, train_count)
val_indices = get_subset(indices, train_count, val_count)

train_loader = torch.utils.data.DataLoader(model_dataset, sampler=SubsetRandomSampler(train_indices),batch_size=batch_size)
val_loader =  torch.utils.data.DataLoader(model_dataset, sampler=SubsetRandomSampler(val_indices),batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)
print(f"Length of Train Data : {len(train_loader)}")
print(f"Length of Validation Data : {len(val_loader)}")
print(f"Length of Test Data : {len(val_loader)}")
print("Detected Classes are: ", model_dataset.class_to_idx)

data_iter = iter(train_loader)
images, labels = data_iter.next()
print('Image minimum and maximum values : ',torch.min(images), torch.max(images))
print('Labels minimum and maximum values : ',torch.min(labels), torch.max(labels))

idx2class = {v: k for k, v in model_dataset.class_to_idx.items()}
idx2class

class Encoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,  stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,  stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,  stride=1, padding=0),
            nn.ReLU(True)
        )
        ### Flatten layer
        self.flatten = nn.Flatten()
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(2 * 2 * 64, 100),
            nn.ReLU(True),
            nn.Linear(100, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
#
model = Encoder(encoded_space_dim=32)

checkpoint = torch.load('/Users/arathi/Desktop/Pycharm/Autoencoder/Lagos/Labeled/PATH/enoder_autoencoder.pth')

model.load_state_dict(checkpoint)
print('Previously trained model weights state_dict loaded...')
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

print("Begin training.")

for e in range(1, num_epochs + 1):

    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch).squeeze()
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # VALIDATION
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch).squeeze()
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            val_epoch_loss += train_loss.item()
            val_epoch_acc += train_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
    print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

# Plot line charts
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",linewidth = 3,  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable",linewidth = 3, ax=axes[1]).set_title('Train-Val Loss/Epoch')
plt.show()

y_pred_list = []
y_true_list = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_test_pred = model(x_batch)
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())

y_pred_list = [i[0] for i in y_pred_list]
y_true_list = [i[0] for i in y_true_list]

print(classification_report(y_true_list, y_pred_list, zero_division=0))

print(confusion_matrix(y_true_list, y_pred_list))

confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=idx2class, index=idx2class)
fig, ax = plt.subplots(figsize=(9,6))
sns.heatmap(confusion_matrix_df, annot=True, ax=ax, fmt="d", cmap='BuGn')
plt.show()

def img_display(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg

dataiter = iter(val_loader)
images, labels = dataiter.next()
class_types = idx2class

# Viewing data examples used for training
fig, axis = plt.subplots(3, 5, figsize=(15, 10))
with torch.no_grad():
    model.eval()
    for ax, image, label in zip(axis.flat,images, labels):
        ax.imshow(img_display(image)) # add image
        image_tensor = image.unsqueeze_(0)
        output_ = model(image_tensor)
        output_ = output_.argmax()
        k = output_.item()==label.item()
        ax.set_title(str(class_types[label.item()])+":" +str(k))
plt.show()