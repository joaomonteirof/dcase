from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from models import resnet

# Training settings
parser = argparse.ArgumentParser(description='Test architectures with dummy data')
parser.add_argument('--model', choices=['resnet'], default='resnet')
parser.add_argument('--nclasses', type=int, default=10, metavar='N', help='number of classes')
args = parser.parse_args()


if args.model == 'resnet':
	model = resnet.ResNet12(n_classes=args.nclasses)

print('\n', model, '\n')
print('\n\nNumber of parameters: {}\n'.format(sum(p.numel() for p in model.parameters())))

batch_1, batch_2 = torch.rand(3, 1, 40, 500), torch.rand(3, 32, 40, 256)

out = model.forward(batch_1, batch_2)

print(out.size(), '\n')