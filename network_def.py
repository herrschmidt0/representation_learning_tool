import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
global Model
import torch.nn as nn

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    ks = 5
    nrf = 16
    I = (125, 80)
    nr_classes = 8

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=nrf, kernel_size=(ks, ks), padding=0)
    self.relu1 = nn.ReLU(inplace=True)
    self.mp1 = nn.MaxPool2d(kernel_size=(2, 2))
    self.dr1 = nn.Dropout(0.5)

    self.conv2 = nn.Conv2d(in_channels=nrf, out_channels=2*nrf, kernel_size=(ks, ks), padding=0)
    self.relu2 = nn.ReLU(inplace=True)
    self.mp2 = nn.MaxPool2d(kernel_size=(2, 1))
    self.dr2 = nn.Dropout(0.5)

    ls = int((nrf/2 * I[0]/4 * I[1]/4 + nr_classes)/2)
    self.dense1 = nn.Linear(1088, ls)
    self.relu3 = nn.ReLU(inplace=True)
    self.dense2 = nn.Linear(ls, nr_classes)


  def forward(self, input):

    x = self.conv1(input)
    x = self.relu1(x)
    x = self.mp1(x)
    x = self.dr1(x) 

    x = self.conv2(x)
    x = self.relu2(x)
    x = self.mp2(x)
    x = self.dr2(x)

    #print(x.shape)
    x = x.reshape(-1, 32*17*2)

    x = self.dense1(x)
    x = self.relu3(x)
    x = self.dense2(x) 

    return x