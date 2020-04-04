from nn import *
from sys import argv, exit
from math import floor
from sparse_svd import sparse_svd
#from jacobian import batch_jacobian
import random

#make the system deterministic
random.seed(1)
torch.manual_seed(1)

def train(net, loader, device, epochs, batch_size, outputs, width, height, hyper_parameter, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epoch_size = floor(height / batch_size)
    # epoch_size * batch_size <= m, orders of 2 may not always be a factor of m
    edjm = torch.zeros(outputs, epoch_size * batch_size, width).to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(loader):
            print(i)
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0].to(device), data[1].to(device, non_blocking=True)

            images.requires_grad_()

            result = net(images).to(device)

            #result.t()[0].backward(torch.ones_like(result.t()[0]), retain_graph=True)

            for o in result.t():
                optimizer.zero_grad()
                #print(o.size())
                o.backward(torch.ones_like(o), retain_graph=True)

                #print(images.grad.size())
                #edjm[0, i*batch_size:(i+1)*batch_size] = images.grad.view(batch_size, width)
                #sparse_svd(images.grad.view(batch_size,width))

            optimizer.zero_grad()
            
            loss = criterion(result, labels)
            loss.backward()
            optimizer.step()

            if i == epoch_size - 1:
                break

        singular_values = 0
        """
        for i in range(outputs):
            singular_values = edjm[i].svd()[1].log().sum()

        optimizer.zero_grad()
        loss = -hyper_parameter*singular_values
        loss.backward(retain_graph=True)
        optimizer.step()
        """
        if verbose:
            print(epoch)
    if verbose:
        print("Finished training")
