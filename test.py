from nn import *
import numpy as np

net = Net()
net.load_state_dict(torch.load(PATH))

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
    
    
print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))

i = 1

for o in outputs.t():
    optimizer.zero_grad()
    o.backward(torch.ones_like(o), retain_graph=True)
    reshaped_grad = images.grad.view(10000, 784)
    print(reshaped_grad.numpy().shape)
    np.savetxt('edjm/edjm' + str(i) + '.csv', reshaped_grad.numpy(), delimiter=',')
    i = i+1
