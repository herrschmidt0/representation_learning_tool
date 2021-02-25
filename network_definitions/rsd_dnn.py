import torch.nn as nn

# Define PyTorch model
class Model(nn.Module):
  def __init__(self):
      super(Model, self).__init__()

      I = 320
      O = 8

      self.fc1 = nn.Linear(in_features=I, out_features=(I+O)//2)
      self.relu1 = nn.ReLU(inplace=True)
      self.fc2 = nn.Linear(in_features=(I+O)//2, out_features=(I+3*O)//4)
      self.relu2 = nn.ReLU(inplace=True)
      self.out = nn.Linear(in_features=(I+3*O)//4, out_features=O)

  def forward(self, x):
      x = self.fc1(x)
      x = self.relu1(x)
      x = self.fc2(x)
      x = self.relu2(x)
      x = self.out(x)
      return x