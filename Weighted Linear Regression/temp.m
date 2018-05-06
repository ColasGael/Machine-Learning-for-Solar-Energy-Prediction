% Best bandwith parameter
tau = 4;

% Prediction
energyTrainPred = ones(mTrain,1);
% Compute the solution for every example x
for i=1: mTrain 
    i
    energyTrainPred(i) = XTrain(i,:)*solution(XTrain(i,:),XTrain,yTrain,tau);
end




% Compute the averaged LMS error
trainAvgLMS = 1/mTrain*sum((yTrain-energyTrainPred).^2);

    %% Weighted linear regression

% Define the closed form solution
    % x : weather example (with intercept term)
    % X : dataset inputs
    % y : dataset outputs
    % tau : bandwith parameter
function theta = solution(x,X,y,tau) 
    w = 1/2*exp(-diag((X-x)*(X-x)')/(2*tau^2));  % define the weight matrix 
    W = diag(w);
    theta = (X'*W*X)\((X')*W*y);               % define the closed form solution
end