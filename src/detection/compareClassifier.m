getStartedDetection;

%
plot_flag = 0;
kfold = 5;
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);

% SVM
learnSVM = @(y, X) trainSVM(y, X, '-t 2 -b 1 -e 0.01');
predictSVM = @(model, X) predictSVM(model, X);
[trAvgTPR_SVM, teAvgTPR_SVM, predTr_SVM, predTe_SVM, trueTr_SVM, trueTe_SVM] = kFoldCrossValidation(y, pcaExpX, kfold, learnSVM, predictSVM, computePerformance, 'SVM RBF kernel');

% Logistic Regression
learnLR = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, 1e-3, [size(X,2) 2]);
predictLR = @(model, X) predictNeuralNetwork(model, X);
[trAvgTPR_LR, teAvgTPR_LR, predTr_LR, predTe_LR, trueTr_LR, trueTe_LR] = kFoldCrossValidation(y, pcaExpX, kfold, learnLR, predictLR, computePerformance, plot_flag, 'Neural Network');

% Neural Network
learnNN = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, 1e-3, [size(X,2) 100 2]);
predictNN = @(model, X) predictNeuralNetwork(model, X);
[trAvgTPR_NN, teAvgTPR_NN, predTr_NN, predTe_NN, trueTr_NN, trueTe_NN] = kFoldCrossValidation(y, X, kfold, learnNN, predictNN, computePerformance, plot_flag, 'Neural Network');

% Random Forest
learnRF = @(y, X) trainRandomForest(y, X, 100, sqrt(size(X,2))/2);
predictRF = @(model, X) predictRandomForest(model, X);
[trAvgTPR_RF, teAvgTPR_RF, predTr_RF, predTe_RF, trueTr_RF, trueTe_RF] = kFoldCrossValidation(y, pcaExpX, kfold, learnRF, predictRF, computePerformance, plot_flag, 'RF');

% Gaussian Proccess classification

[pcaX50,~, ~] = pcaApplyOnData(X, PCA.coeff, PCA.mu, 50);
% Normalize PCA features
[pcaX50, ~, ~] = zscore(pcaX50);

learnGP = @(y, X) trainGPClassification(y, X);
predictGP = @(model, X) predictGPClassification(model,X);
[trAvgTPR_GP, teAvgTPR_GP, predTr_GP, predTe_GP, trueTr_GP, trueTe_GP] = kFoldCrossValidation(y, pcaX50, kfold, learnGP, predictGP, computePerformance, plot_flag, 'GP');


%%

methodNames = {'Log Reg', 'NN', 'GP', 'RF', 'SVM'};
avgTPRList = kCVevaluateMultipleMethods( cat(3, trueTe_LR, trueTe_NN, trueTe_GP, trueTe_RF, trueTe_SVM), cat(3, predTe_LR, predTe_NN, predTe_GP, predTe_RF, predTe_SVM), true, methodNames );
