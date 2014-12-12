%% Get and prepare data

getStartedDetection;

%% Feature transformations


%% kCV

plot_flag = 1;
learn = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, 1.e-4);
predict = @(model, X) predictNeuralNetwork(model, X);
computePerformance = @(trueOutputs, pred, model_name) kCVfastROC(trueOutputs, pred, model_name, 1);
[trAvgTPR, teAvgTPR, predTr, predTe] = kFoldCrossValidation(y, X, 3, learn, predict, computePerformance, 'Logistic Regression');

trAvgTPR
teAvgTPR

%TODO: kCV on other models
%% Learn parameters using kCV

dpFractions = [0.45 0.5 0.55 0.6];
%wDecays = [1e-5 1e-4 1e-3 1e-2 1e-1];
[bestDp, trainTPR, testTPR] = findDropoutNeuralNetwork(y, pcaX, 3, dpFractions, 1);

% bestDp: 0.2000 bestWd: 1.0000e-04

%% Learn parameters using kCV

dpFractions = [0.45 0.5 0.55 0.6];
%wDecays = [1e-5 1e-4 1e-3 1e-2 1e-1];
[bestDp, trainTPR, testTPR] = findDropoutNeuralNetwork(y, pcaX, 3, dpFractions, 1);

% bestDp: 0.2000 bestWd: 1.0000e-04

%% Test results

% Methods names for legend
methodNames = {'regular','square','square root','Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( Te.y > 0, [rfPred, randPred], true, methodNames );