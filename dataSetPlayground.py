import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np

# function to show an image using matplotlib from a tensor


def show(img):
    # convert tensor to numpy array
    npimg = img.numpy()
    # Convert to H*W*C shape
    npimg_tr = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg_tr, interpolation='nearest')


# path to store or retrieve data
data_path = './data'

# loading training data set
train_data = datasets.MNIST(data_path, train=True, download=True)

# extracting data and labels
train_x = train_data.data
train_y = train_data.targets
print(train_x.shape)
print(train_y.shape)

# loading test data
test_data = datasets.MNIST(data_path, train=False, download=True)

# loading test data
test_x = test_data.data
test_y = test_data.targets
print(test_x.shape)
print(test_y.shape)

# adding a dimension to the data
if len(train_x.shape) == 3:
    train_x = train_x.unsqueeze(1)

print(train_x.shape)

if len(test_x.shape) == 3:
    test_x = test_x.unsqueeze(1)

print(test_x.shape)

grid_x = utils.make_grid(train_x[:40], nrow=8, padding=2)
print(grid_x.shape)

show(grid_x)
