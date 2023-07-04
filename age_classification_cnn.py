import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import shutil
import os
from google.colab import drive

dataset_path = './modelsd'

drive.mount('/content/drive')
source_path = '/content/drive/MyDrive/CNN/RI'

shutil.move(source_path, dataset_path)

training_dataset = torchvision.datasets.CelebA(dataset_path, split='train', target_type='attr', download=True)
validation_dataset = torchvision.datasets.CelebA(dataset_path, split='valid', target_type='attr', download=True)
testing_dataset = torchvision.datasets.CelebA(dataset_path, split='test', target_type='attr', download=True)

print('Training set:', len(training_dataset))
print('Validation set:', len(validation_dataset))
print('Testing set:', len(testing_dataset ))


training_transform = transforms.Compose([
    transforms.RandomCrop([178, 178]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAutocontrast(),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.CenterCrop([178, 178]),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])


all_attributes = training_dataset.attr_names
index_of_young = all_attributes.index("Young")
print("Index of 'young' attribute:", index_of_young)


extract_young = lambda attributes: attributes[39]

training_dataset = torchvision.datasets.CelebA(dataset_path, split='train', target_type='attr', download=False,
                                                transform=training_transform, target_transform=extract_young)
validation_dataset = torchvision.datasets.CelebA(dataset_path, split='valid', target_type='attr', download=False,
                                                 transform=transform, target_transform=extract_young)
testing_dataset = torchvision.datasets.CelebA(dataset_path, split='test', target_type='attr', download=False,
                                                  transform=transform, target_transform=extract_young)


training_dataset = Subset(training_dataset, torch.arange(26000))
validation_dataset = Subset(validation_dataset, torch.arange(7200))
testing_dataset  = Subset(testing_dataset , torch.arange(3648))

print('Training set:', len(training_dataset))
print('Validation set:', len(validation_dataset))
print('Testing set:', len(testing_dataset ))


batch_size = 32

training_data_loader = DataLoader(training_dataset, batch_size, shuffle=True)
validation_data_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
testing_data_loader = DataLoader(testing_dataset, batch_size, shuffle=False)


model = nn.Sequential()

model.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1))
model.add_module('batchnorm1', nn.BatchNorm2d(32))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout1', nn.Dropout(p=0.6))

model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1))
model.add_module('batchnorm2', nn.BatchNorm2d(64))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout2', nn.Dropout(p=0.4))

model.add_module('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1))
model.add_module('batchnorm3', nn.BatchNorm2d(128))
model.add_module('relu3', nn.ReLU())
model.add_module('pool3', nn.MaxPool2d(kernel_size=2))

model.add_module('conv4', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=1))
model.add_module('batchnorm4', nn.BatchNorm2d(256))
model.add_module('relu4', nn.ReLU())

model.add_module('pool4', nn.AvgPool2d(kernel_size=4, padding=0))
model.add_module('flatten', nn.Flatten())

model.add_module('fc', nn.Linear(256, 1))
model.add_module('sigmoid', nn.Sigmoid())


device = torch.device("cuda:0")
model = model.to(device)


loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def save_model(model, path):
  drive.mount('/content/drive', force_remount=True)
  directory = '/content/drive/MyDrive/SavedModels'
  os.makedirs(directory, exist_ok=True)
  torch.save(model, os.path.join(directory, path))

def load_model():
  drive.mount('/content/drive', force_remount=True)
  directory = '/content/drive/MyDrive/SavedModels/first-try.ph'
  loaded_model = torch.load(directory)
  return loaded_model


def train(model, num_epochs, training_data_loader, validation_data_loader):
    training_loss_hist, training_accuracy_hist, validation_loss_hist, validation_accuracy_hist = [], [], [], []
    for i in range(num_epochs):
        training_loss, training_accuracy, validation_loss, validation_accuracy = 0, 0, 0, 0

        model.train()
        for x_batch, y_batch in training_data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(x_batch)[:, 0]
            loss = loss_fn(prediction, y_batch.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()*y_batch.size(0)
            is_prediction_correct = ((prediction>=0.5).float() == y_batch).float()
            training_accuracy += is_prediction_correct.sum().cpu()

        training_loss /= len(training_data_loader.dataset)
        training_accuracy /= len(training_data_loader.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in validation_data_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                prediction = model(x_batch)[:, 0]
                loss = loss_fn(prediction, y_batch.float())
                validation_loss += loss.item()*y_batch.size(0)
                is_prediction_correct = ((prediction>=0.5).float() == y_batch).float()
                validation_accuracy += is_prediction_correct.sum().cpu()

        validation_loss /= len(validation_data_loader.dataset)
        validation_accuracy /= len(validation_data_loader.dataset)

        training_loss_hist.append(training_loss)
        training_accuracy_hist.append(training_accuracy)
        validation_loss_hist.append(validation_loss)
        validation_accuracy_hist.append(validation_accuracy)

        print(f'Epoch {i+1} train accuracy: {training_accuracy:.4f} validation accuracy: {validation_accuracy:.4f}')


    x_arr = np.arange(len(training_loss_hist)) + 1

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, training_loss_hist, '-o', label='Train loss')
    ax.plot(x_arr, validation_loss_hist, '--<', label='Validation loss')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, training_accuracy_hist, '-o', label='Train acc.')
    ax.plot(x_arr, validation_accuracy_hist, '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)


def test(model):
  testing_accuracy = 0

  model.eval()
  with torch.no_grad():
      for x_batch, y_batch in testing_data_loader:
          x_batch = x_batch.to(device)
          y_batch = y_batch.to(device)
          prediction = model(x_batch)[:, 0]
          is_prediction_correct = ((prediction>=0.5).float() == y_batch).float()
          testing_accuracy += is_prediction_correct.sum().cpu()

  testing_accuracy /= len(testing_data_loader.dataset)
  print(f'Test accuracy: {testing_accuracy:.4f}')
  visualize_test_results(x_batch, y_batch)


def visualize_test_results(x_batch_for_visualization, y_batch_for_visualization):
    prediction = model(x_batch_for_visualization)[:, 0] * 100
    fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(15, 20), gridspec_kw={'width_ratios': [0.3, 0.7]})

    for j in range(0, 10):

        ax_image = axes[j, 1]
        ax_text = axes[j, 0]

        ax_image.set_xticks([])
        ax_image.set_yticks([])
        ax_image.imshow(x_batch_for_visualization[j].cpu().permute(1, 2, 0))

        label = "Young" if y_batch_for_visualization[j] == 1 else "Old"
        prediction_percentage = f"{prediction[j]:.0f}%"
        classified_as = "Young" if prediction[j] >= 50 else "Old"
        text = f"Ground truth: {label:s}\nCNN classification: {classified_as}\n with the percentage of youngness {prediction_percentage}"
        ax_text.text(0.5, 0.5, text, size=16, ha="center", va="center")
        ax_text.axis("off")

    plt.tight_layout(pad=0)
    plt.show()

torch.manual_seed(1)
num_epochs = 30
train(model, num_epochs, training_data_loader, validation_data_loader)
test(model)
save_model(model, "first-try.ph")
