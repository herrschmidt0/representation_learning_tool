class Model(nn.Module):
	
	IMAGE_SIZE = 256
	NUM_CHANNELS = 1
	NUM_LABELS = 2

	FILTER_SIZE = 5
	FILTER_1 = 50
	FILTER_2 = 30
	HIDDEN_NUM = 32
	LEARNING_RATE = 0.01

	def __init__(self):
		super(Model, self).__init__()

		self.conv1 = nn.Conv2d(self.NUM_CHANNELS, self.FILTER_1, kernel_size=self.FILTER_SIZE, padding=2)
		self.conv2 = nn.Conv2d(self.FILTER_1, self.FILTER_2, kernel_size=self.FILTER_SIZE, padding=2)
		#self.conv3 = nn.Conv2d(self.FILTER_2, self.FILTER_2, kernel_size=self.FILTER_SIZE, padding=2)

		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(self.IMAGE_SIZE//16 * self.IMAGE_SIZE//16 * self.FILTER_2, self.NUM_LABELS)
		self.fc2 = nn.Linear(self.HIDDEN_NUM, self.NUM_LABELS)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 4))
		x = F.relu(F.max_pool2d(self.conv2(x), 4))
		#x = F.relu(F.max_pool2d(self.conv3(x), 2))

		#x = x.view(-1, self.IMAGE_SIZE//4 * self.IMAGE_SIZE//4 * self.FILTER_2)
		x = self.flatten(x)
		#x = F.relu(self.fc1(x))
		x = self.fc1(x)
		return x

