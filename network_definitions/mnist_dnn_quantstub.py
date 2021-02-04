from torch.quantization import QuantStub, DeQuantStub

input_size = 784
hidden_size = 500
num_classes = 10

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.quant = QuantStub()  
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out = self.dequant(x)
        return out