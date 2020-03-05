function [S] = sparse_svd(X)
    sizeX = size(X{1});
    m = sizeX(1);
    n = sizeX(2);
    k = floor(n/10);
    p = 10;
    l = k+p;

    Phi = randi([0 1],l,m);
    S = cell(1,length(X));

    % algorithm
    for i=1:length(X)
        Y = Phi*X{i};
        B = Y*Y';
        B = 0.5*(B + B');
        [T,D] = eigs(B,k);
        St = sqrt(D);
        Vt = Y'*T/St;
        Ut = X{i}*Vt;
        S{i} = svd(Ut);
    end
end