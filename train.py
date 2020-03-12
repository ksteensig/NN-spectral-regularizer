from nn import *
from sys import argv, exit
from math import floor
from sparse_svd import sparse_svd
from jacobian import batch_jacobian
import random

#make the system deterministic
#random.seed(1)
#torch.manual_seed(1)



def train(net, loader, device, epochs, batch_size, outputs, width, height, hyper_parameter, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epoch_size = floor(height / batch_size)
    # epoch_size * batch_size <= m, orders of 2 may not always be a factor of m
    edjm = torch.zeros(outputs, epoch_size * batch_size, width).to(device)

    

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            edjm[:outputs,
                 i * batch_size:(i + 1) * batch_size] = batch_jacobian(
                     net, images,
                     outputs, device).view(batch_size, outputs,
                                   width).permute(1, 0, 2).requires_grad_()

            optimizer.zero_grad()

            result = net(images)

            loss = criterion(result, labels)
            loss.backward()
            optimizer.step()

            if i == epoch_size - 1:
                break

        singular_values = 0

        for i in range(outputs):
            singular_values = edjm[i].svd()[1].log().sum()

        optimizer.zero_grad()
        loss = -hyper_parameter*singular_values
        loss.backward(retain_graph=True)
        optimizer.step()

        if verbose:
            print(epoch)
    if verbose:
        print("Finished training")

"""
# train network
for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        print(i)

        optimizer.zero_grad()

        edjm[:k, i * batch_size:(i + 1) * batch_size] = get_batch_jacobian(
            net, inputs, 10).view(128, 10, 784).permute(1, 0,
                                                        2).requires_grad_()

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i == epoch_size - 1:
            break

    singular_values = 0

    for i in range(k):
        singular_values = sparse_svd(edjm[i])[1].log().sum()

    optimizer.zero_grad()
    loss = -singular_values
    loss.backward(retain_graph=True)
    optimizer.step()

    print(epoch)
"""
