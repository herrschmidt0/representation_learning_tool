import torchvision
import torchvision.transforms as transforms

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
	}
}