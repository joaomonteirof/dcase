from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from models import vgg, resnet, densenet, base_cnn, TDNN

# Training settings
parser = argparse.ArgumentParser(description='Test architectures with dummy data')
parser.add_argument('--model', choices=['cnn', 'vgg', 'resnet', 'densenet', 'tdnn'], default='resnet')
parser.add_argument('--nclasses', type=int, default=10, metavar='N', help='number of classes')
args = parser.parse_args()


if args.model == 'cnn':
	model = base_cnn.CNN(n_classes=args.nclasses)
elif args.model == 'vgg':
	model = vgg.VGG('VGG16', n_classes=args.nclasses)
elif args.model == 'resnet':
	model = resnet.ResNet50(n_classes=args.nclasses)
elif args.model == 'densenet':
	model = densenet.DenseNet121(n_classes=args.nclasses)
elif args.model == 'tdnn':
	model = TDNN.TDNN(n_classes=args.nclasses)

print('\n', model, '\n')
print('\n\nNumber of parameters: {}\n'.format(sum(p.numel() for p in model.parameters())))

batch = torch.rand(3, 1, 40, 129)

out = model.forward(batch)

print(out.size(), '\n')