
from importlib.util import set_loader
import torch
from torch import nn
from torch.nn import functional as F

def cifar_cnn(num_classes):
    model = CNN(num_classes)
    return model

from torch.nn.utils import weight_norm


# Make a CNN & train it to predict genres.

class CNN(nn.Module):
  def __init__(self, num_classes=10):
    """Intitalize neural net layers"""
    super(CNN, self).__init__()
    self.weights = 128
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0)
    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0)
    self.fc1 = nn.Linear(in_features=128, out_features=num_classes) #in_features=128,

    self.batchnorm1 = nn.BatchNorm2d(num_features=8)
    self.batchnorm2 = nn.BatchNorm2d(num_features=16)
    self.batchnorm3 = nn.BatchNorm2d(num_features=32)
    self.batchnorm4 = nn.BatchNorm2d(num_features=64)
    self.batchnorm5 = nn.BatchNorm2d(num_features=128)

    self.dropout = nn.Dropout(p=0.3, inplace=False)


  def forward(self, x):
    # Conv layer 1.
    #print(x.size())
    x = self.conv1(x)
    x = self.batchnorm1(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 2.
    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 3.
    x = self.conv3(x)
    x = self.batchnorm3(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)
    #x = F.max_pool2d(x, kernel_size = 2)
    #x = F.max_pool2d(x, kernel_size = 2)
    

    '''# Conv layer 4.
    x = self.conv4(x)
    x = self.batchnorm4(x)
    x = F.relu(x)
    print(x.size())
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 5.
    x = self.conv5(x)
    x = self.batchnorm5(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2) #)'''

    # Fully connected layer 1.
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    x = self.dropout(x)
    
    self.weights = nn.Parameter(x)
    
    y = torch.zeros(13, x.size()[1]).cuda()
    y[:x.size()[0],:x.size()[1]] = self.weights
    
    x = F.linear(x, y)
    
    x = F.softmax(x)
    

    return x,None


