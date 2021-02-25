import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

def train(model, args):

    train_loader = args[0]
    device = args[1]
    num_epochs = args[2]
    learning_rate = 0.001 

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 8 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    return model