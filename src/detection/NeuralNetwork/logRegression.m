getStartedDetection;

%% Try on full or reduced data

plot_flag = 0;
predict = @(model, X) predictNeuralNetwork(model, X);
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);

learn = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, 0, [size(X,2),2]);
[trAvgTPR_pcaX, teAvgTPR_pcaX, predTr_pcaX, predTe_pcaX, trueTr_pcaX, trueTe_pcaX] = kFoldCrossValidation(y, pcaX, 3, learn, predict, computePerformance, plot_flag, 'Neural Network');

[trAvgTPR_pcaExpX, teAvgTPR_pcaExpX, predTr_pcaExpX, predTe_pcaExpX, trueTr_pcaExpX, trueTe_pcaExpX] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, plot_flag, 'Neural Network');

[trAvgTPR_X, teAvgTPR_X, predTr_X, predTe_X, trueTr_X, trueTe_X] = kFoldCrossValidation(y, X, 3, learn, predict, computePerformance, plot_flag, 'Neural Network');

% Prediction performances on the two different models
methodNames = {'pca(X)', 'pca(exp(X))','X'};
avgTPRst = kCVevaluateMultipleMethods( cat(3, trueTe_pcaX, trueTe_pcaExpX, trueTe_X), cat(3, predTe_pcaX, predTe_pcaExpX, predTe_X), true, methodNames );
%savePlot('./report/figures/detection/pca-varianceExplained.pdf','Principal Components','Variance Explained');

% We have better results using the data after PCA (on both test and train)
%% Try weight decay (dropout is not applicable since just 1 layer with 2 outputs )

weightValues = [0 1e-5 1e-4 1e-3 1e-2];
[bestWD, trainTPRlog, testTPRlog] = findWeightDecayNeuralNetwork(y, pcaExpX, 3, weightValues, 1);
%savePlot('./report/figures/detection/logreg-regularization-learningcurve.pdf','Weight Decay on L2','TPR on train (blue) and test (red)');

% Best weight value is 1e-3 for which we have a test TPR of: 0.8285
% trainTPRlog reveals the power of regularization: the model fits less the
% training data and get improvement on generalizing on test data!