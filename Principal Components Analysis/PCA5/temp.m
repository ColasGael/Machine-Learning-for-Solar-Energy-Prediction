
% Plot the error for each point on a X-time
figure();
    % Compute the LMS error for each point
[timeAxis,I] = sort(datenum(timeline));             % sort the test example dates
testLMSerror = (yTest(I)-energyTestPred(I)).^2;     % sort the test examples according to their date
plot(timeAxis,testLMSerror);
datetick('x',12);
xlabel('Example date');
ylabel('LMS prediction error');

    % Compute the absolute relative error for each non zero (y=0) point
J = find(yTest(I) ~= 0); yTestSort = yTest(I); energyTestPredSort = energyTestPred(I);
testARerror = abs(yTestSort(J)-energyTestPredSort(J))./yTestSort(J);
plot(timeAxis(J),testARerror);
datetick('x',12);
xlabel('Example date');
ylabel('Absolute relative prediction error');

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
