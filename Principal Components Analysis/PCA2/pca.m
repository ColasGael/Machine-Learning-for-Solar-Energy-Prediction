%% Principal Component Analysis

function Z = pca(X,k)
    % Normalize the data
    mu = mean(X);   % mean of the data
    s = std(X,1);   % standard deviation of each coordinate
    X_norm = (X-mu)./s;

    % Compute the empirical covariance matrix of the data
    m = size(X,1);
    Sig = 1/m*(X_norm'*X_norm);

    % Compute the eigenvector of Sig
    [V,D] = eig(Sig);
    [~,I] = sort(diag(abs(D)),'descend');    % sort the eigenvalues
    Comp = V(:,I);

    % Compute the k principal components
    U = Comp(:,1:k);
    
    % Compute the k-dimensional representation of the data
    Z = X*U;
end