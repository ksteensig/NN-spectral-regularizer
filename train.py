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

            for j in range(outputs):
                #pass
                optimizer.zero_grad()
                result.t()[j].backward(torch.ones_like(result.t()[j]), retain_graph=True)
                edjm[j, i*batch_size:(i+1)*batch_size] = images.grad.view(batch_size, width).requires_grad_()

            optimizer.zero_grad()
            
            loss = criterion(result, labels)
            loss.backward()
            optimizer.step()

            if i == epoch_size - 1:
                break

        if regularizer:
            #pass
            #X = torch.randn(10, 10000, 784).to(device).requires_grad_()
            #U,S,V = batch_svd(X)
            #loss = U.sum() + S.log().sum() + V.sum()
            #loss.backward()
            U,S,V = batch_svd(edjm.detach().requires_grad_())
            optimizer.zero_grad()
            loss = -hyper_parameter*S.log().sum()
            loss.backward()
            optimizer.step()

        if (epoch-1) % 5 == 0:
            old_loss = new_loss
            new_loss = validation(net, device, criterion, validationloader)
            if new_loss < 0.35:
                break
            print('epoch ' + str(epoch) + ' has validation loss of ' + str(new_loss))

        epoch = epoch + 1

    #print("complexity score of: " + str(calc_score(edjm)))

    print("Finished training")


def validation(net, device, loss_fun, validationloader):
    loss = 0
    for i, data in enumerate(validationloader):
        images, labels = data[0].to(device), data[1].to(device, non_blocking=True)
        result = net(images).to(device)

        loss = loss + loss_fun(result, labels)

    return loss.item()

def calc_score(edjm):
    edjm.requires_grad = True
    s = 0
    for i in range(10):
        U,S,V = edjm[i].svd()
        s = s + S[0:74].sum().mul(S[0].pow(-1))
    return s.mul(0.1)

