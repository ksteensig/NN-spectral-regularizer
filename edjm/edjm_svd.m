edjm1 = csvread('~/Workspace/faster-neural-network-training/edjm/edjm1.csv');
edjm2 = csvread('~/Workspace/faster-neural-network-training/edjm/edjm2.csv');
edjm3 = csvread('~/Workspace/faster-neural-network-training/edjm/edjm3.csv');
edjm4 = csvread('~/Workspace/faster-neural-network-training/edjm/edjm4.csv');
edjm5 = csvread('~/Workspace/faster-neural-network-training/edjm/edjm5.csv');
edjm6 = csvread('~/Workspace/faster-neural-network-training/edjm/edjm6.csv');
edjm7 = csvread('~/Workspace/faster-neural-network-training/edjm/edjm7.csv');
edjm8 = csvread('~/Workspace/faster-neural-network-training/edjm/edjm8.csv');
edjm9 = csvread('~/Workspace/faster-neural-network-training/edjm/edjm9.csv');
edjm10 = csvread('~/Workspace/faster-neural-network-training/edjm/edjm10.csv');

%%
S(1:784,1) = svd(edjm1);
S(1:784,2) = svd(edjm2);
S(1:784,3) = svd(edjm3);
S(1:784,4) = svd(edjm4);
S(1:784,5) = svd(edjm5);
S(1:784,6) = svd(edjm6);
S(1:784,7) = svd(edjm7);
S(1:784,8) = svd(edjm8);
S(1:784,9) = svd(edjm9);
S(1:784,10) = svd(edjm10);

%%
figure(1)
hold on;
plot(sum(S')/10)
plot(S(:,1))
plot(S(:,2))
plot(S(:,3))
plot(S(:,4))
plot(S(:,5))
plot(S(:,6))
plot(S(:,7))
plot(S(:,8))
plot(S(:,9))
plot(S(:,10))

%%
figure(2)
hold on;
plot(S1./max(S1))
plot(S2./max(S2))
plot(S3./max(S3))
plot(S4./max(S4))
plot(S5./max(S5))
plot(S6./max(S6))
plot(S7./max(S7))
plot(S8./max(S8))
plot(S9./max(S9))
plot(S10./max(S10))