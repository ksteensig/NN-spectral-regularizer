import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class FFNet(nn.Module):
    def __init__(self, inputs, outputs, layer_units, layer_count, path):
        super(FFNet, self).__init__()
        self.path = path

        self.layers = nn.ModuleList([nn.Linear(inputs, layer_units)])
        self.layers.extend([nn.Linear(layer_units, layer_units) for i in range(1, layer_count-1)])
        self.layers.append(nn.Linear(layer_units, outputs))

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))

        for l in self.layers[:-1]:
            x = F.relu(l(x))
        
        return self.layers[-1](x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
