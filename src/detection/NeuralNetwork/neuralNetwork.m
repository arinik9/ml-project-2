getStartedDetection;

%% Try out on X or pcaX

plot_flag = 0;
predict = @(model, X) predictNeuralNetwork(model, X);
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);

learn = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, 0);
[trAvgTPR_pcaX, teAvgTPR_pcaX, predTr_pcaX, predTe_pcaX, trueTr_pcaX, trueTe_pcaX] = kFoldCrossValidation(y, pcaX, 3, learn, predict, computePerformance, plot_flag, 'Neural Network');

[trAvgTPR_X, teAvgTPR_X, predTr_X, predTe_X, trueTr_X, trueTe_X] = kFoldCrossValidation(y, X, 3, learn, predict, computePerformance, plot_flag, 'Neural Network');

% Prediction performances on the two different models
methodNames = {'pca(X)', 'X'};
avgTPRList = kCVevaluateMultipleMethods( cat(3, trueTe_pcaX, trueTe_X), cat(3, predTe_pcaX, predTe_X), true, methodNames );
%savePlot('./report/figures/detection/pca-varianceExplained.pdf','Principal Components','Variance Explained');


%% Try out different activation functions

plot_flag = 0;
predict = @(model, X) predictNeuralNetwork(model, X);
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);

learn_sigm = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, 0);
[trAvgTPR_sigm, teAvgTPR_sigm, predTr_sigm, predTe_sigm, trueTr_sigm, trueTe_sigm] = kFoldCrossValidation(y, X, 3, learn_sigm, predict, computePerformance, plot_flag, 'Sigmoid Activation');

learn_tanh = @(y, X) trainNeuralNetwork(y, X, 0, 2, 'tanh_opt', 0, 0);
[trAvgTPR_tanh, teAvgTPR_tanh, predTr_tanh, predTe_tanh, trueTr_tanh, trueTe_tanh] = kFoldCrossValidation(y, X, 3, learn_tanh, predict, computePerformance, plot_flag, 'Tanh Activation');


% Prediction performances on the two different models
methodNames = {'Tanh', 'Sigmoid'};
avgTPRList = kCVevaluateMultipleMethods( cat(3, trueTe_tanh, trueTe_sigm), cat(3, predTe_tanh, predTe_sigm), true, methodNames );

% Sigmoid provide better results (improvement of 10% on avgTPR) and is more
% robust (smaller variance). Plot?
%% Try out different number of activation functions
%TODO: write a function to find parameters automatically

activationRange = [10 20 50 100];
[bestNactivation, trainTPRact, testTPRact] = findActivationsNeuralNetwork(y, X, 3, activationRange);
savePlot('./report/figures/detection/nn-activation-learningcurve-fulldata-bis.pdf','Activation function number on L2','Train (blue) and test (red) TPR');

% Best activation number seems to be 11
% We can see that with too many activations function on the second layer
% that we are clearly overfitting the training data and therefore we are doing
% worse on the test data

%% Find the best regularization

dpFractions = [0 0.2 0.3 0.4 0.5];
weightDecays = [0 1e-4 1e-3 1e-2];
[bestDropout, bestWeight, trainTPR_dw, testTPR_dw] = findParamsNeuralNetwork(y, X, 3, dpFractions, weightDecays);
savePlot('./report/figures/detection/nn-regu-learningcurve-fulldata.pdf','Drop out fraction','TPR on test');

% bestDropout: 0.3, bestWeight: 1e-03