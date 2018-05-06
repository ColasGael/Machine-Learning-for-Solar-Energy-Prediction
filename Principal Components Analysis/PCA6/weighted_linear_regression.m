%% WEIGHTED LINEAR REGRESSION WITH PCA
clear all; 
close all;

%% Load weather data
load weather_train.csv;
XTrain = weather_train(:, 2:end-1); % weather examples (input)
yTrain = weather_train(:,end);      % energy examples (output)
load weather_dev.csv;
XDev = weather_dev(:, 2:end-1);     % weather examples (input)
yDev = weather_dev(:,end);          % energy examples (output)
load weather_test.csv;
XTest = weather_test(:, 2:end-1);   % weather examples (input)
yTest = weather_test(:,end);        % energy examples (output)

load weather_test_timeline.csv;     
timeline = string(weather_test_timeline(:,1)) + "-" + string(weather_test_timeline(:,2)) + "-" + string(weather_test_timeline(:,3));
timeline = datetime(timeline,'Format','dd-MM-yyyy'); % the dates of every example of the test set

n = size(XTrain,2);                 % nb of weather features
mTrain = size(yTrain,1);            % nb of examples of the train set
mDev = size(yDev,1);                % nb of examples of the dev set
mTest = size(yTest,1);              % nb of examples of the test set

%% Apply PCA over the whole data set
% Complete data set
X = [XTrain; XDev; XTest];

% Compressed data set
k = 6; % nb of dimensions of the compressed data set
Z = pca(X,k);

% Split the compressed data set
Z_train = Z(1:mTrain,:);
Z_dev = Z(mTrain+1:mTrain+mDev,:);
Z_test = Z(mTrain+mDev+1:mTrain+mDev+mTest,:);

%% Weighted linear regression with PCA

% Define the data set (adding the intercept term) 
Z_train = [ones(mTrain,1) Z_train];
Z_dev = [ones(mDev,1) Z_dev];
Z_test = [ones(mTest,1) Z_test];

    %% Hold out cross validation

% Define the different bandwidth parameters
Tau = [2 3 4 10 50 100];
% Define the legend for each tau
tauLegend = strings(size(Tau));
tauLegend(:) = "tau = " ;
tauLegend = tauLegend + string(Tau);

hold on;
% Plot the raw dev set
XAxis = (1:1:mDev);
plot(XAxis,yDev);

% Dev error
devError = ones(size(Tau',1),1);
for k=1 : size(Tau')
    % Define the closed form solution
        % the solution theta(x,tau) is defined at the end of the file
    
    tau = Tau(k);
    % Prediction
    energyDevPred = ones(mDev,1);
    % Compute the solution for every example x
    for i=1: mDev 
        i
        energyDevPred(i) = Z_dev(i,:)*solution(Z_dev(i,:),Z_train,yTrain,tau);
    end
    % Plot the prediction
    plot(XAxis,energyDevPred);
    % Compute the error
    devError(k) = 1/mDev*sum((yDev-energyDevPred).^2);
end

% Print the legend
xlabel('Example number');
ylabel('Energy kW.h^{-1}');
plotLegend = ["dev set" tauLegend];
legend(plotLegend);
% Save the results
print('solar-energy_weighted-linear-regression', '-dpng')
hold off;

% Plot the learning curve
figure();
plot((1:1:size(Tau,2)),devError);
xlabel('Bandwith parameter number');
ylabel('LMS prediction error');
print('weighted-linear-regression_bandwidth-parameter_learning-curve', '-dpng')

    %% Chosen model

% Best bandwith parameter
tau = 3;

figure(); 
hold on;
% Plot the raw dev set
t = 100;
XAxis = (1:1:mTest);
plot(XAxis(1:t),yTest(1:t));
% Prediction
energyTestPred = ones(mTest,1);
% Compute the solution for every example x
for i=1: mTest 
    i
    energyTestPred(i) = Z_test(i,:)*solution(Z_test(i,:),Z_train,yTrain,tau);
end
% Plot the prediction for t test examples
plot(XAxis(1:t),energyTestPred(1:t));
xlabel('Example number');
ylabel('Energy kW.h^{-1}');
legend('Test set', 'WLR prediction');
print('solar-energy_weighted-linear-regression_chosen-tau_test-set', '-dpng');
hold off;

% Plot the error for each point on a X-time
figure();
    % Compute the LMS error for each point
[timeAxis,I] = sort(datenum(timeline));             % sort the test example dates
testLMSerror = (yTest(I)-energyTestPred(I)).^2;     % sort the test examples according to their date
plot(timeAxis,testLMSerror);
datetick('x',12);
xlabel('Example date');
ylabel('LMS prediction error');
print('solar-energy_weighted-linear-regression_test-LMS-error', '-dpng');

    % Compute the absolute relative error for each non zero (y=0) point
J = find(yTest(I) ~= 0); yTestSort = yTest(I); energyTestPredSort = energyTestPred(I);
testARerror = abs(yTestSort(J)-energyTestPredSort(J))./yTestSort(J);
plot(timeAxis(J),testARerror);
datetick('x',12);
xlabel('Example date');
ylabel('Absolute relative prediction error');
print('solar-energy_weighted-linear-regression_test-AR-error', '-dpng');

% Compute the averaged LMS error
testAvgLMS = 1/mTest*sum((yTest-energyTestPred).^2);
% Compute the median absolute relative error
testMedAR = median(testARerror);

    % Compute the absolute relative error for point at hour = 8 ; 12 ; 16
hours = weather_test(:,1); hoursSort = hours(I);
days = weather_test_timeline(:,1); daysSort = days(I);
months = weather_test_timeline(:,2); monthsSort = months(I);
years = weather_test_timeline(:,3); yearsSort = years(I);

K8 = find(yTest(I)~=0 & hours(I)==8);
testARerror8 = abs(yTestSort(K8)-energyTestPredSort(K8))./yTestSort(K8);
M8 = [hoursSort(K8) daysSort(K8) monthsSort(K8) yearsSort(K8) testARerror8];

K12 = find(yTest(I)~=0 & hours(I)==12);
testARerror12 = abs(yTestSort(K12)-energyTestPredSort(K12))./yTestSort(K12);
M12 = [hoursSort(K12) daysSort(K12) monthsSort(K12) yearsSort(K12) testARerror12];

K16 = find(yTest(I)~=0 & hours(I)==16);
testARerror16 = abs(yTestSort(K16)-energyTestPredSort(K16))./yTestSort(K16);
M16 = [hoursSort(K16) daysSort(K16) monthsSort(K16) yearsSort(K16) testARerror16];


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