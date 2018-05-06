%% Weighted linear regression
clear all; 
close all;

% Load weather data
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

% Define the data set (adding the intercept term) 
XTrain = [ones(mTrain,1) XTrain];
XDev = [ones(mDev,1) XDev];
XTest = [ones(mTest,1) XTest];

    %% Hold out cross validation

% Define the different bandwidth parameters
Tau = [3 4 5 6 10 50 100 1000];
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
        energyDevPred(i) = XDev(i,:)*solution(XDev(i,:),XTrain,yTrain,tau);
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
tau = 4;

% Prediction
energyTestPred = ones(mTest,1);
% Compute the solution for every example x
for i=1: mTest 
    i
    energyTestPred(i) = XTest(i,:)*solution(XTest(i,:),XTrain,yTrain,tau);
end

% Plot the error for each point on a X-time axis
figure();
    % Compute the LMS error for each point
[timeAxis,I] = sort(datenum(timeline)); % sort the test example dates
yTestSort = yTest(I);                   % sort the test examples according to their date
energyTestPredSort = energyTestPred(I);

testLMSerror = (yTestSort-energyTestPredSort).^2;     
plot(timeAxis,testLMSerror);
datetick('x',12);
xlabel('Example date');
ylabel('LMS prediction error');
print('solar-energy_weighted-linear-regression_test-LMS-error', '-dpng');

    % Compute the absolute relative error for each non zero (y=0) point
J = find(yTest(I) ~= 0);
testARerror = abs(yTestSort(J)-energyTestPredSort(J))./yTestSort(J);
plot(timeAxis(J),testARerror);
datetick('x',12);
xlabel('Example date');
ylabel('Absolute relative prediction error');
print('solar-energy_weighted-linear-regression_test-AR-error', '-dpng');

    % Compute the absolute relative error for non winter point
hours = weather_test(:,1); hoursSort = hours(I);
days = weather_test_timeline(:,1); daysSort = days(I);
months = weather_test_timeline(:,2); monthsSort = months(I);
years = weather_test_timeline(:,3); yearsSort = years(I);
K = find(yTestSort ~= 0 & monthsSort ~=1 & monthsSort ~=2 & monthsSort ~=3 & monthsSort ~=4);
testARerrorCut = abs(yTestSort(K)-energyTestPredSort(K))./yTestSort(K);
plot(timeAxis(K),testARerrorCut);
datetick('x',12);
xlabel('Example date');
ylabel('Absolute relative prediction error');
print('solar-energy_WLR_test-AR-error_non-winter', '-dpng');
testMedARCut = median(testARerrorCut);

% Compute the averaged LMS error
testAvgLMS = 1/mTest*sum((yTest-energyTestPred).^2);
% Compute the median absolute relative error
testMedAR = median(testARerror);

% Plot the prediction for one year examples on a X-time axis
figure(); 
hold on;
L = find(yearsSort==2016 & hoursSort>9 & hoursSort<15 & monthsSort~=1 & monthsSort~=2 & monthsSort~=3 & monthsSort~=4);
yTestShort = yTest(L);
energyTestPredShort = energyTestPred(L);
timeAxisShort = timeAxis(L);         
    % Plot the raw dev set
plot(timeAxisShort,yTestShort);
plot(timeAxisShort,energyTestPredShort);
datetick('x',12);
xlabel('Example date');
ylabel('Energy kW.h^{-1}');
legend('Test set', 'WLR prediction');
print('solar-energy_WLR_prediction-comparison_test-set', '-dpng');
hold off;

    % Compute the absolute relative error for point at hour = 8 ; 12 ; 16
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