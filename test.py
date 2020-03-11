from nn import *
import numpy as np

net = FFNet(784, 10, 3072, 2, 'mnist_2_3072')

net.load_state_dict(torch.load(net.path + '.pth'))

testset = torchvision.datasets.MNIST(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=10000,
                                         shuffle=False)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

correct = 0
total = 0
images = None

for data in testloader:
    images, labels = data
    images.requires_grad_()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %lf %%' %
      (100 * correct / total))
