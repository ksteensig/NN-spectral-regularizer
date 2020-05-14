from nn import *
from torch_batch_ops import batch_svd 

def test(net, loader):
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    svd = torch.zeros(10, 78*2)

    for i, data in enumerate(loader):
        images, labels = data[0].to('cuda:0'), data[1].to('cuda:0')
        images.requires_grad_()
        outputs = net(images).to('cuda:0')
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        s = 0
        
        for i in range(10):
            optimizer.zero_grad()
            outputs.t()[i].backward(torch.ones_like(outputs.t()[i]), retain_graph=True)
            reshaped_grad = images.grad.view(10000, 784)
            U,S,V = reshaped_grad.svd()
            s = s + S[0:78*2].sum().mul(S[0].pow(-1))
            #svd[0] = S[0:78]

        print(s/10)
        

    return 100 * correct / total
