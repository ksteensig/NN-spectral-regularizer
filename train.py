from nn import *

trainset = torchvision.datasets.MNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          pin_memory=True)

net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device, non_blocking=True)

print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train network
for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        print(inputs.size())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(epoch)

torch.save(net.state_dict(), PATH + '.pth')

print('Finished Training')
