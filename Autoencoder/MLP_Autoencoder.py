import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset
import os
from natsort import natsorted
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 24
epochs = 100
lr = 1e-3
d = 2
NOISE_FACTOR = 0.5

source1 = '/home/ubuntu/Capstone/Autoencoder/Satellite_Images/Train'

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



transform = transforms.Compose([transforms.ToTensor()])

dataset = CustomDataSet(source1, transform=transform)

# indices = torch.randperm(len(dataset))[:500]
# dataloader = torch.utils.data.DataLoader(dataset,sampler=SubsetRandomSampler(indices),batch_size=batch_size)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(next(iter(dataloader)).shape)
dataiter = iter(dataloader)
images = dataiter.next()
print(torch.min(images), torch.max(images))



class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=300, out_features=50),
            nn.ReLU(True),
            nn.Linear(in_features=50, out_features=25),
            nn.BatchNorm1d(25),
            nn.ReLU(True),
            nn.Linear(in_features=25, out_features=16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=8),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Linear(in_features=8, out_features=2),
            )
    def forward(self, x):
        x = x.view(-1, 3 * 10 * 10)
        x = self.encoder(x)
        return x

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=8),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Linear(in_features=8, out_features=16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=25),
            nn.BatchNorm1d(25),
            nn.ReLU(True),
            nn.Linear(in_features=25, out_features=50),
            nn.BatchNorm1d(50),
            nn.ReLU(True),
            nn.Linear(in_features=50, out_features=300)
        )

    def forward(self, x):
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
encoder = Encoder()
decoder = Decoder()
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params_to_optimize, lr=lr,weight_decay=1e-5)

def add_noise(inputs, NOISE_FACTOR):
    noisy = inputs + torch.randn_like(inputs) * NOISE_FACTOR
    noisy = torch.clip(noisy, 0., 1.)
    return noisy

print('Cuda available',torch.cuda.is_available())

def training(dataloader, num_epochs):

    train_loss = []

    for epoch in range(1, num_epochs + 1):

        encoder.train()
        decoder.train()

        # monitor training loss
        runing_loss= 0.0

        # Training
        for data  in  dataloader:
            img = data
            img = img.to(device)

            image_noisy = add_noise(img, NOISE_FACTOR)
            image_noisy = image_noisy.to(device)
            # ===================forward=====================
            encoded_data = encoder(image_noisy)
            decoded_data = decoder(encoded_data)
            loss = criterion(decoded_data, img.view(-1, 3 * 10 * 10))
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

        torch.save(encoder.state_dict(), '/home/ubuntu/Autoencoder/Autoencoder/Path/mlp_enoder_autoencoder.pth')
        torch.save(decoder.state_dict(), '/home/ubuntu/Autoencoder/Autoencoder/Path/mlp_decoder_autoencoder.pth')

    return train_loss

train_loss = training(dataloader, epochs)

plt.figure()
plt.plot(train_loss)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('/home/ubuntu/Autoencoder/Autoencoder/Path/mlp_train_ae__loss.png')


def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

#Batch of test images
dataiter = iter(dataloader)
images = dataiter.next()

#Sample outputs
encoded_data = encoder(images)
decoded_data = decoder(encoded_data)
images = images.numpy()

output = decoded_data.view(batch_size, 3, 10, 10)
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