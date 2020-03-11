from nn import *
from sys import argv, exit
from math import floor
from sparse_svd import sparse_svd

batch_size = 128
m = 10000  # height of the EDJM matrices (amount of samples used for training)
N = 784  # width of the EDJM matrices (size of an MNIST image vector)
k = 10  # number of outputs/EDJMs

epoch_size = floor(10000 / 128)

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device, non_blocking=True)

print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# epoch_size * batch_size <= m, orders of 2 may not always be a factor of m
edjm = torch.zeros(k, epoch_size * batch_size, N).to(device)


def get_batch_jacobian(net, x, to):
    # noutputs: total output dim (e.g. net(x).shape(b,1,4,4) noutputs=1*4*4
    # b: batch
    # i: in_dim
    # o: out_dim
    # ti: total input dim
    # to: total output dim
    x_batch = x.shape[0]
    x_shape = x.shape[1:]
    x = x.unsqueeze(1)  # b, 1 ,i
    x = x.repeat(1, to, *(1, ) * len(x.shape[2:]))  # b * to,i  copy to o dim
    x.requires_grad_(True)
    tmp_shape = x.shape
    y = net(x.reshape(-1,
                      *tmp_shape[2:]))  # x.shape = b*to,i y.shape = b*to,to
    y_shape = y.shape[1:]  # y.shape = b*to,to
    y = y.reshape(x_batch, to, to)  # y.shape = b,to,to
    input_val = torch.eye(to).reshape(1, to, to).repeat(x_batch, 1, 1).to(
        device)  # input_val.shape = b,to,to  value is (eye)
    y.backward(input_val, retain_graph=True)  # y.shape = b,to,to
    return x.grad.reshape(x_batch, *y_shape, *x_shape).data  # x.shape = b,o,


# train network
for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        edjm[:k, i * batch_size:(i + 1) * batch_size] = get_batch_jacobian(
            net, inputs, 10).view(128, 10, 784).permute(1, 0, 2).requires_grad_()

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

torch.save(net.state_dict(), net.path + '.pth')

print('Finished Training')
