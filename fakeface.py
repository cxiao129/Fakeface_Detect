from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import json
from PIL import Image
from tqdm import tqdm


# Set random seed for reproducibility
manualSeed = random.randint(1, 10000)     # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

workers = 1
batch_size = 32
image_size = 200
num_epochs = 30
lr = 0.0002
beta1 = 0.5
ngpu = 0


# define database class
class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])

        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

# Split into training set and validation set
def read_split_data(root: str, val_rate: float = 0.75):      # modify the val_rate to change the rate of validation
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)


    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    classes.sort()

    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in classes:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))


    return train_images_path, train_images_label, val_images_path, val_images_label


train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(r"/home/peer/Desktop/虚假人脸检测/CNN_synth_testset")

data_transform = {
    "train": transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}


train_dataset = MyDataSet(images_path=train_images_path,
                            images_class=train_images_label,
                            transform=data_transform["train"])

val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label,
                        transform=data_transform["val"])

# Create the dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


class Net(nn.Module):      
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

        self.fc1 = nn.Linear(50 * 50 * 16, 128)    
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):     
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      

        x = x.view(x.size()[0], -1)     
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)      

# Create the Discriminator
netD = Net().to(device)


# Print the model
print(netD)

# Train
optimizer = optim.Adam(netD.parameters(), lr=lr)    
criterion = nn.CrossEntropyLoss()

def train(model, optimizer, data_loader, device):
    for epoch in range(num_epochs):
        data_loader = tqdm(data_loader)
        val_acc = 0.0
        val_loss = 0.0

        for step, data in enumerate(data_loader):
            image, label = data
            optimizer.zero_grad()
            image,  label = image.to(device), label.to(device)
            # print(label)
            pred = model(image)
            loss = criterion(pred, label.squeeze())  
            loss.backward()
            optimizer.step()
        
        val_acc, val_loss = evaluate(netD, val_loader, device)
        print('epoch = {:>2d}, val_loss = {:>.5f}, val_acc = {:.3%}'.format(epoch + 1, val_loss, val_acc))


def evaluate(model, data_loader, device):
        correct_num = 0.0
        total_loss = 0.0
        for step, data in enumerate(data_loader):
            image, label = data
            image,  label = image.to(device), label.to(device)
            with torch.no_grad():
                pred = model(image)
                total_loss += criterion(pred, label.squeeze()).item()     
                correct_num += (pred.argmax(1) == label ).type(torch.float).sum().item()

        val_loss = total_loss / len(data_loader)
        val_acc = correct_num / len(data_loader.dataset) 
        return  val_acc, val_loss

train(netD, optimizer, train_loader, device)