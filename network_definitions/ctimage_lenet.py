class Model(nn.Module):
	
	IMAGE_SIZE = 512
	NUM_CHANNELS = 1
	PIXEL_DEPTH = 255
	NUM_LABELS = 2

	FILTER_SIZE = 5
	FILTER_1 = 20
	FILTER_2 = 50
	HIDDEN_NUM = 100
	LEARNING_RATE = 0.01

	def __init__(self):
		super(Model, self).__init__()

		self.conv1 = nn.Conv2d(self.NUM_CHANNELS, self.FILTER_1, kernel_size=self.FILTER_SIZE, padding=2)
		self.conv2 = nn.Conv2d(self.FILTER_1, self.FILTER_2, kernel_size=self.FILTER_SIZE, padding=2)

		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(self.IMAGE_SIZE//4 * self.IMAGE_SIZE//4 * self.FILTER_2, self.HIDDEN_NUM)
		self.fc2 = nn.Linear(self.HIDDEN_NUM, self.NUM_LABELS)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))

		x = x.view(-1, self.IMAGE_SIZE//4 * self.IMAGE_SIZE//4 * self.FILTER_2)
		#x = self.flatten(x)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

