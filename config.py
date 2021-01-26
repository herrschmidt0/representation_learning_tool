import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

config = {
	"mnist-1d": {
		"datasets": [
			torchvision.datasets.MNIST(root='../../data', 
                                       train=True, 
                                       transform=transforms.ToTensor(),  
                                       download=True), 
			torchvision.datasets.MNIST(root='../../data', 
                                       train=False, 
                                       transform=transforms.ToTensor())
		],
		"transform": lambda x: x.reshape(-1, 28*28),
		
		"network-def": "network_definitions/mnist_dnn_pytorch.py",
		"train": "trainers/mnist_dnn_train_pytorch.py",
		"weights": "exported_models/dnn_mnist.pth"
	},
	"mnist-2d": {
		"datasets": [
			torchvision.datasets.MNIST(root='../../data', 
                                       train=True, 
                                       transform=transforms.ToTensor(),  
                                       download=True), 
			torchvision.datasets.MNIST(root='../../data', 
                                       train=False, 
                                       transform=transforms.ToTensor())
		],
		"transform": lambda x: x,
		
		"network-def": "network_definitions/mnist_cnn_pytorch.py",
		"train": "trainers/mnist_dnn_train_pytorch.py",
		"weights": "exported_models/cnn_mnist.pt"
	},
	"ct-medical-images": {
		"datasets": 'dataset_loaders/ctimage_loader.py',
		"transform": lambda x: x, 
		"network-def": "network_definitions/ctimage_lenet.py",
		"train": "trainers/ctimage_train.py"
	}
}