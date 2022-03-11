import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from natsort import natsorted
import os
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image
import torchvision


source1 = r'/home/ubuntu/Capstone/Autoencoder/Lagos/Labeled/PNG/'

num_epochs = 100
batch_size = 32
learning_rate = 1e-3

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image.to(device,torch.float)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = CustomDataSet(source1, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset , batch_size=batch_size, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(next(iter(dataloader)).shape)
dataiter = iter(dataloader)
images = dataiter.next()
print(torch.min(images), torch.max(images))

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential( # Formula - (W-F +2P)/S + 1

        # conv 1 - input (10-3 + 2*0)/1 + 1
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0), # 10x10 --> 8x8
        nn.BatchNorm2d(16),
        nn.ReLU(),

        # conv 2 (8-5 + 2*0)/1 + 1
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0), #4x4
        nn.BatchNorm2d(32),
        nn.ReLU(),

        # conv 3 (4-3 + 2*0)/1 + 1
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),  # 2x2
        nn.BatchNorm2d(64),
        )
        self.decoder = nn.Sequential( # s * (W-1) + F - 2P
        # conv 4 (2 * 2-1 + 5 - 2*2)
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=3, padding=2),#output_padding=1),#4x4
        nn.BatchNorm2d(32),
        nn.ReLU(),
        # conv 5 (2 * 4-1 + 5 - 2*2)
        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=3, padding=3),#8x8
        nn.BatchNorm2d(16),
        # conv 6 (1 * 8-1 + 3 - 2*0)
        nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=0,),#10x10
        nn.Tanh()

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder()
print(model)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = get_device()
print(device)
model.to(device)

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 10, 10)
    return x

def save_decod_img(img, epoch):
    img = img.view(img.size(0), 3, 10, 10)
    save_image(img, '/home/ubuntu/Capstone/Autoencoder/Lagos/Labeled/Model_Images/Autoencoder_image{}.png'.format(epoch))

def training(model, dataloader, num_epochs):
    train_loss = []
    for epoch in range(1, num_epochs + 1):
        # monitor training loss
        runing_loss= 0.0

        # Training
        for data in dataloader:
            img = data
            img = img.to(device)
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runing_loss += loss.item() * img.size(0)
            # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss))

        loss = runing_loss / len(dataloader)
        train_loss.append(loss)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, loss))
        if epoch % 10 == 0:
            save_decod_img(output.cpu().data, epoch)

        torch.save(model.state_dict(), '/home/ubuntu/Capstone/Autoencoder/Lagos/Labeled/Path/conv_autoencoder.pth')
    return train_loss

device = get_device()
model.to(device)

train_loss = training(model, dataloader, num_epochs)
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss',)
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig('/home/ubuntu/Capstone/Autoencoder/Lagos/Labeled/train_ae__loss.png')

def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

#Batch of test images
dataiter = iter(dataloader)
images = dataiter.next()

#Sample outputs
output = model(images)
images = images.numpy()

output = output.view(batch_size, 3, 10, 10)
output = output.detach().numpy()

#Original Images
print("Original Images")
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    #ax.set_title(classes[labels[idx]])
plt.show()

#Reconstructed Images
print('Reconstructed Images')
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    #ax.set_title(classes[labels[idx]])
plt.show()