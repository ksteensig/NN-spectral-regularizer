from nn import *
from sys import argv, exit
from math import floor
from torch_batch_ops import batch_svd
from test import test


def train(net, trainloader, validationloader, device, batch_size, outputs, width, height, hyper_parameter, regularizer=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epoch_size = floor(height / batch_size)
    # epoch_size * batch_size <= m, orders of 2 may not always be a factor of m
    edjm = torch.zeros(outputs, epoch_size * batch_size, width).to(device)

    eps = -10e-5
    old_loss = 1001.0
    new_loss = 1000.0

    epoch = 1

    while old_loss > new_loss:
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0].to(device), data[1].to(device, non_blocking=True)

            images.requires_grad_()

            result = net(images).to(device)

            for o in result.t():
                optimizer.zero_grad()
                o.backward(torch.ones_like(o), retain_graph=True)

            optimizer.zero_grad()
            
            loss = criterion(result, labels)
            loss.backward()
            optimizer.step()

            if i == epoch_size - 1:
                break

        if regularizer:
            edjm.requires_grad = True
            U,S,V = batch_svd(edjm)
            optimizer.zero_grad()
            loss = -hyper_parameter*S.log().sum()
            loss.backward(retain_graph=True)
            optimizer.step()

        if (epoch-1) % 5 == 0:
            old_loss = new_loss
            new_loss = validation(net, device, criterion, validationloader)
            print('epoch ' + str(epoch) + ' has validation loss of ' + str(new_loss))

        epoch = epoch + 1

    print("Finished training")


def validation(net, device, loss_fun, validationloader):
    loss = 0
    for i, data in enumerate(validationloader):
        images, labels = data[0].to(device), data[1].to(device, non_blocking=True)
        result = net(images).to(device)

        loss = loss + loss_fun(result, labels)

    return loss.item()
