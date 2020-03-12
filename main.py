from nn import *
from sys import argv, exit
from math import floor
from sparse_svd import sparse_svd
from jacobian import batch_jacobian
import random

from train import train
from test import test

epochs = 20
batch_size = 128
height = 10000  # height of the EDJM matrices (amount of samples used for training)
width = 784  # width of the EDJM matrices (size of an MNIST image vector)
outputs = 10  # number of outputs/EDJMs

optimize_hyper_parameter = True
do_train = False
do_test = False

hyper_parameter = 0.1

# handle configuration inputs, exit if there are not enough
if len(argv) < 6:
    print('Incorrect parameters: type inputs outputs layer_count layer_units')
    exit()

net_type = argv[1]
net_inputs = int(argv[2])
net_outputs = int(argv[3])
net_layer_count = int(argv[4])
net_layer_units = int(argv[5])

path = net_type + '_' + str(net_layer_count) + '_' + str(net_layer_units)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = FFNet(net_inputs, net_outputs, net_layer_units, net_layer_count, path)

# set up data, this should ideally be done based on the net_type
trainset = torchvision.datasets.MNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          pin_memory=True,
                                          shuffle=False)

testset = torchvision.datasets.MNIST(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=10000,
                                         shuffle=True)

net.to(device, non_blocking=True)

print(net)

if optimize_hyper_parameter:
    old_result = 0
    best_hp = 0
    for i in range(11):
        hp = i * 0.1
        net = FFNet(net_inputs, net_outputs, net_layer_units, net_layer_count,
                    path)
        train(net, trainloader, device, epochs, batch_size, outputs, width,
              height, hp, True)

        new_result = test(net, testloader)

        if new_result > old_result:
            old_result = new_result
            best_hp = hp

    print('best lambda: ' + str(best_hp))
        

if do_train:
    train(net, trainloader, device, epochs, batch_size, outputs, width, height,
          hyper_parameter, True)
    torch.save(net.state_dict(), path + '.pth')

if do_test:
    net.load_state_dict(torch.load(path + '.pth'))
    print(test(net, testloader))
