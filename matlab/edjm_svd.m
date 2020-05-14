mnist_path = {'mnist_1_6144', 'mnist_2_3072', 'mnist_3_2048', 'mnist_4_1536'};

mnist_data = {};

for i=1:length(mnist_path)
    path = mnist_path{i};
    
    for j=1:10
        mnist_data{j,i} = csvread(strcat('~/Workspace/faster-neural-network-training/', path, '/', path, num2str(j), '.csv'));
    end
end


%%

svd_collection = {};

for i=1:4
    for j=1:10
        svd_collection{j,i} = svd(mnist_data{j,i});
    end
end

%%

%figure(1)
%hold on;

score = zeros(1,4);
svds = zeros(784,4);

for i=1:length(score)
    for j=1:10
        s = svd_collection{j,i};
        m = max(s);
        score(i) = score(i) + sum(s(1:78)/m);
        svds(:, i) = svds(:, i) + s;
    end
end

score = score./10
svds = svds./10;

%% plot
plot(svds(1:74,:))
legend('1 layer, 6144 units','2 layers, 3072 units','3 layers, 2048 units','4 layers, 1536 units')
xlabel('Singular value order')
ylabel('Value')

%%

mse_svd = zeros(10,4);

for i=1:4
    for j=1:10
        [u,s,v] = svd(mnist_data{j,i});
        [U,S,V] = sparse_svd(mnist_data{j,i});
        delta = (diag(S) -  diag(s(1:78,1:78)));
        mse_svd(j,i) = sum(delta.^2)/length(diag(s));
    end
end