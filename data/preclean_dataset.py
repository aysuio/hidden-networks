import torch
import torchvision
import numpy as np

def load_data_cifar(imbal_size, original_size):
"""
original array size is 5000
if imbal_size = 500/original_size=1000, it generates [500, 1000] * 5
"""
# Load CIFAR10
dataset = torchvision.datasets.CIFAR10(
    root='./',
    transform=torchvision.transforms.ToTensor(),
    download=True)

# Get all training targets and count the number of class instances
targets = np.array(dataset.targets) # 长度为10000的list，对应包含data中每一张图片的label
classes, class_counts = np.unique(targets, return_counts=True)
nb_classes = len(classes)
print(class_counts)

# Create artificial imbalanced class counts
imbal_class_counts = [imbal_size, original_size] * 5
# Get class indices
class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

# Get imbalanced number of instances
imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
imbal_class_indices = np.hstack(imbal_class_indices)    # https://stackoverflow.com/questions/57047417/numpy-hstack-description

# Set target and data to dataset
dataset.targets = targets[imbal_class_indices]
dataset.data = dataset.data[imbal_class_indices]

assert len(dataset.data) == len(dataset.targets)