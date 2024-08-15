import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

rgb = 3
greyscale = 1
h, w = 256, 256

input_dim=(h,w)
channel_dim=greyscale

# Paths to the data
train_path = "data/training/"
test_path = "data/testing/"
validation_path = "data/validation/"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

class_names = [name[len(train_path):] for name in glob.glob(f'{train_path}*')]
class_names = dict(zip(range(len(class_names)), class_names))
class_names

class CustomDataset(Dataset):
    def __init__(self, img_size, class_names, path=None, transformations=None, num_per_class: int = -1, channel_dim='rgb'):
        self.img_size = img_size
        self.path = path
        self.num_per_class = num_per_class
        self.class_names = class_names
        self.transforms = transformations
        self.channel_dim = channel_dim
        self.data = []
        self.labels = []

        if path:
            self.readImages()

        # Default transforms applied to the testing and validation data
        self.standard_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def readImages(self):
        for id, class_name in self.class_names.items():
            print(f'Loading images from class: {id} : {class_name}')
            img_path = glob.glob(os.path.join(self.path, class_name, '*.jpg'))
            if self.num_per_class > 0:
                img_path = img_path[:self.num_per_class]
            self.labels.extend([id] * len(img_path))
            for filename in img_path:
                if self.channel_dim == rgb:
                    img = Image.open(filename).convert('RGB')
                else:
                    img = Image.open(filename).convert('L')
                self.data.append(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transforms:
            img = self.transforms(img)  # Apply the custom transforms, including augmentations
        else:
            img = self.standard_transforms(img)  # Apply standard transforms if no custom ones are provided

        label = torch.tensor(label, dtype=torch.long)

        return img, label
    
# Define the data augmentation transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(15),      # Randomly rotate the image by 15 degrees
    transforms.RandomResizedCrop(input_dim),  # Random crop followed by resizing
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness, contrast, etc.
    transforms.ToTensor(),              # Convert the image to a tensor
])

# Create an instance of CustomDataset with augmentations
train_dataset = CustomDataset(
    img_size=input_dim,
    class_names=class_names,
    path=train_path,
    transformations=train_transform,
    channel_dim=channel_dim
)

train_dataloader = DataLoader(train_dataset, shuffle=True)

validation_dataset = CustomDataset(
    img_size=input_dim,
    class_names=class_names,
    path=validation_path,
    channel_dim=channel_dim
)

validation_dataloader = DataLoader(validation_dataset, shuffle=True)

test_dataset = CustomDataset(
    img_size=input_dim,
    class_names=class_names,
    path=test_path,
    channel_dim=channel_dim
)

test_dataloader = DataLoader(test_dataset, shuffle=True)

class group_9(nn.Module):
    def __init__(self):
        super(group_9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Correct the input size for fc1 based on the calculations
        self.fc1 = nn.Linear(64 * 125*125, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return self.softmax(x)
    
    
model = group_9()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

model.to(device)


def train(model, num_epochs: int = 3):
    training_array, validation_array = [], []
    trainloss_array, validationloss_array = [],[]
    guessed, target_labels = [],[]
    
    for epoch in range(num_epochs):
        guessed = []
        target_labels = []
        training_correct, training_total, training_loss = 0, 0, 0

        model.train()
        length = 0
        for data, targets in train_dataloader:
            length += 1
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            guessed.append(predicted.item())
            target_labels.append(targets.item())
            training_total += targets.size(0)
            training_correct += (predicted == targets).sum().item()
            training_accuracy = training_correct / training_total * 100
            training_loss += outputs.shape[0] * loss.item()
            print(training_loss)

        print(f'Training accuracy: {training_accuracy}%')
        training_array.append(training_accuracy)
        trainloss_array.append(training_loss/length)
        print(trainloss_array)


        model.eval()
        
        correct_validation, total_validation = 0, 0
        with torch.no_grad():
            for data, targets in validation_dataloader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, targets)
                total_validation += targets.size(0)
                correct_validation += (predicted == targets).sum().item()

        validation_accuracy = 100 * correct_validation / total_validation
        
        validation_array.append(validation_accuracy)
        validationloss_array.append(loss.item() * outputs.shape[0])
        
        print(f'Validation accuracy: {validation_accuracy}%')
        torch.save(model.state_dict(), f'./models/model_tune_hyper_epoch_{epoch}.pth')
    return np.array(training_array), np.array(validation_array), np.array(trainloss_array), np.array(validationloss_array)
        
def test(model):
    model.eval()
    correct = 0
    total = 0
    predicted_labels, actual_labels = [], []
    with torch.no_grad():
        for data, targets in test_dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            predicted_labels.append(predicted)
            actual_labels.append(targets.item())
    accuracy = 100 * correct / total
    return accuracy, predicted_labels, actual_labels

train_acc_arr, validation_acc_arr = train(model=model, num_epochs=3)
test_acc, predicted_labels, actual_labels = test(model=model)
print(f"Test accuracy: {test_acc:.2f}%")

torch.save(model.state_dict(), 'group_9.pth')


def plot_metric(metric_name, training_array, validation_array):
    """
    Plots the given metric for both training and validation data using Matplotlib.
    
    Parameters:
    - metric_name: The name of the metric to plot (e.g., 'loss' or 'accuracy').
    - training_array: array of the loss/accuracy values for each epoch collected from training set.
    - validation_array: array of the loss/accuracy values for each epoch collected from validation set.
    """
    
                    
    num_epochs = len(training_array) #getting the amount of epoches from an input array
    epochs = range(1, num_epochs + 1) #sequence of epoch numbers for the plot
    plt.plot(epochs, training_array, 'b', label=f'Training {metric_name}')
    plt.plot(epochs, validation_array, 'g', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}' )
    plt.xlabel('Epochs')
    plt.ylabel(f' {metric_name}')
    plt.legend()

    plt.show()

def show_confusion_matrix(actual_labels, predicted_labels):
    """
    This function prints the confusion matrix.
    :param actual_labels: array of 1(label 'Pneumonia') and 0(label 'Healthy') given with dataset.
    :param predicted_labels: array of 1(label 'Pneumonia') and 0(label 'Healthy') predicted by model.
    """
    cm = confusion_matrix(actual_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Healthy', 'Pneumonia'], 
            yticklabels=['Healthy', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    

    
Training_data = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34]) ## TODO: INPUT REAL NUMBERS
Validation_data = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1]) ## TODO: INPUT REAL NUMBERS

epochs = range(1, 10) ## TODO: INPUT REAL NUMBERS

plot_metric('Loss', Training_data, Validation_data)
plot_metric('Accuracy', Training_data, Validation_data)
show_confusion_matrix(actual_labels, predicted_labels)