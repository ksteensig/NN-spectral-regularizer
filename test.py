from nn import *


def test(net, loader):
    correct = 0
    total = 0

    for i, data in enumerate(loader):
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total
