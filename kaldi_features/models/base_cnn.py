import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

CFG = [32, 'M', 64, 'M']

class CNN(nn.Module):
	def __init__(self, n_classes=1000):
		super(CNN, self).__init__()

		self.n_classes = n_classes

		self.features = self._make_layers(CFG)
		self.classifier = nn.Sequential(nn.Linear(128, 64),
										nn.Dropout(0.3),
										nn.ReLU(),
										nn.Linear(64, self.n_classes) )

	def forward(self, x):
		x = self.features(x).mean(-1)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)

		return x

	def _make_layers(self, cfg):
		layers = []
		in_channels = 1
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=5, stride=2, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=2, stride=1)]
		return nn.Sequential(*layers)