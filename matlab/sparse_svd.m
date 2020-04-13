function [U,S,V] = sparse_svd(X)
    sz = size(X);
    m = sz(1);
    n = sz(2);
    k = floor(n/10);
    p = 10;
    l = k + p;

    %index = randsample(1:length(X), l);
    %Y = X(index, :);
    
    Phi = randn(l, m);
    Y = Phi*X;
    
    B= Y*Y';
    B = 0.5*(B + B');    
    [T,D] = eigs(B,k);
    St = sqrt(D);
    Vt = Y'*T/St;
    Ut = X*Vt;
    [U,S,Q] = svd(Ut, 'econ');
    %S = diag(S);
    V = Vt*Q';
end