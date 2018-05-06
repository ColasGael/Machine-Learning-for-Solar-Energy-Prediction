%% PCA empirical correlation

function C = pcaCor(X,k)
    % Normalize the data
    mu = mean(X);   % mean of the data
    s = std(X,1);   % standard deviation of each coordinate
    size(s)
    X_norm = (X-mu)./s;

    % Compute the empirical covariance matrix of the data
    m = size(X,1);
    Sig = 1/m*(X_norm'*X_norm);

    % Compute the eigenvector of Sig
    [V,D] = eig(Sig);
    [~,I] = sort(diag(abs(D)),'descend');    % sort the eigenvalues
    Comp = V(:,I);
    
    absEig = diag(abs(D));
    eigSort = absEig(I);

    % Compute the k principal components
    U = Comp(:,1:k);
    
    V = abs(U)./sum(abs(U));
    W = 1/sum(eigSort(1:k))*V.*eigSort(1:k)';
    % Compute the empirical correlation of the data
    C = sum(W,2);
end