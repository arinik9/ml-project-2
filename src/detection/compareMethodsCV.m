%% Get and prepare data

getStartedDetection;

%% Feature transformations

%% kCV

plot_flag = 1;
learn = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, 1.e-4);
predict = @(model, X) predictNeuralNetwork(model, X);
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);
[trAvgTPR, teAvgTPR, predTr, predTe, trueTr, trueTe] = kFoldCrossValidation(y, pcaX, 3, learn, predict, computePerformance, plot_flag, 'Logistic Regression');

%%

dpFractions = [0.45 0.9];
trainModel = @(y, X, param) trainNeuralNetwork(y, X, 0, 1, 'sigm', param, 0, [size(X,2),2]);
predictModel = @(model, X) predictNeuralNetwork(model, X);
[dp, trTPR, teTPR] = find1Param(y, pcaX, 2, trainModel, predictModel, dpFractions, 1);


%%

% Methods names for legend
methodNames = {'Train', 'coucou', 'test'};

% Prediction performances on different models
avgTPRList = kCVevaluateMultipleMethods( cat(3, trueTr(1:500,1:2), trueTe(1:500,1:2), trueTe(1:500,1:2)), cat(3,predTr(1:500,1:2), predTe(1:500,1:2), predTr(1:500,1:2)), true, methodNames );
avgTPRList
%TODO: kCV on other models

%%

methodNames = {'Model','Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( trueTe(:,1) > 0, [predTe(:,1), predTr(1:size(predTe,1),1)], true, methodNames );


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