% Script used to test different Random Forest settings
% First we find which dataset we should work with, then we look for the
% optimal number of trees needed and finally we try to play with other
% parameters such as the minimum observation per leaf and the number of
% variables to select at random for each decision split.

getStartedDetection;

%% Model

% Parameters to play with:
% - nTrees: Number of trees in the random forest
% - 'NVarToSample': Number of variables to select at random for each decision split. 
%                 Default is the square root of the number of variables for classification
% - 'MinLeaf': Minimum number of observations per tree leaf. Default is 1 for classification
% - 'FBoot': Fraction of input data to sample with replacement from the input data for growing each new tree.
%            set 'SampleWithReplacement' to off to play with it

%% Test on pca dataset or full dataset

plot_flag = 0;
learn = @(y, X) trainRandomForest(y, X, 100);
predict = @(model, X) predictRandomForest(model, X);
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);

fprintf('Random forest on pca(X)\n')
[trAvgTPR_pcaX, teAvgTPR_pcaX, predTr_pcaX, predTe_pcaX, trueTr_pcaX, trueTe_pcaX] = kFoldCrossValidation(y, pcaX, 3, learn, predict, computePerformance, plot_flag, 'Random Forest');

fprintf('Random forest on pca(exp(X))\n')
[trAvgTPR_pcaExpX, teAvgTPR_pcaExpX, predTr_pcaExpX, predTe_pcaExpX, trueTr_pcaExpX, trueTe_pcaExpX] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, plot_flag, 'Random Forest');

fprintf('Random forest on X\n')
[trAvgTPR_X, teAvgTPR_X, predTr_X, predTe_X, trueTr_X, trueTe_X] = kFoldCrossValidation(y, X, 3, learn, predict, computePerformance, plot_flag, 'Random Forest');


% Prediction performances on the three different models
methodNames = {'pca(X)', 'pca(exp(X))', 'X'};
avgTPRList = kCVevaluateMultipleMethods( cat(3, trueTe_pcaX, trueTe_pcaExpX, trueTe_X), cat(3, predTe_pcaX, predTe_pcaExpX, predTe_X), true, methodNames );

% Working with pca(exp(X)) provide better results
%% Learning: Number of trees in the forest
nTreesValues = [50 100 200 300 400 500];
[bestnTree, trainTPR, testTPR] = findnTreesRF(y, pcaExpX, 3, nTreesValues, 1);
savePlot('./report/figures/detection/rf-nbtrees-learningcurve.pdf','Number of trees','Train (blue) and test (red) TPR');

% 100 gives good results as well and is a good compromise between computation complexity and performance
%% Learning: Number of features for bagging

baggingRange = [size(pcaExpX,2), 5*sqrt(size(pcaExpX,2)), 2*sqrt(size(pcaExpX,2)), sqrt(size(pcaExpX,2)), sqrt(size(pcaExpX,2)) / 2, sqrt(size(pcaExpX,2)) / 5];
[bestVarSample, trainTPR, testTPR] = findnVarSampleRF(y, pcaExpX, 3, baggingRange, 1);
savePlot('./report/figures/detection/rf-nbvarsample-learningcurve.pdf','Features to sample','Train (blue) and test (red) TPR');

% sqrt(size(pcaExpX,2)) / 2 gives us the best result
%% Learning: MinLeaf value

leafRange = [1, 10, 50, 100, 500];
[bestVarSample, trainTPR, testTPR] = findminLeafRF(y, pcaExpX, 3, leafRange, 1);
savePlot('./report/figures/detection/rf-minleaf-learningcurve.pdf','Minimum observation per leaf','Train (blue) and test (red) TPR');

% 1 gives us the best result