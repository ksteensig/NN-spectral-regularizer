import torch

def batch_jacobian(net, x, to, device):
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
