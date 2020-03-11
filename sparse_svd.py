import math
import torch

# only calculate the singular values, U and V don't matter for this project
# X is a matrix of dimensions (m,n) where m>=n
# the function makes a low rank approximation (10% of n) for the singular values
def sparse_svd(X):
    size = list(X.size())
    m = size[0]
    n = size[1]

    p = 10
    k = math.floor(n/10)
    l = k+p # estimate a low rank approx that is 10% of  with p oversampling

    Phi = torch.empty(l, m).random_(2)
    Y = Phi.mm(X)

    B = Y.mm(Y.t())
    B = B.add(B.t()).mul_(0.5)

    D,T = torch.symeig(B, eigenvectors=True);
    print(D.diag().size())
    print(T.size())
    St = D.diag()[:k,:k].pow(0.5)
    Vt = Y.t().mm(T[:,:k]).mm(St.inverse())
    Ut = X.mm(Vt)

    return Ut.svd(compute_uv=True)
