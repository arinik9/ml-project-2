% make predictions
clearvars;

%% Load data and normalize

fprintf('Load data..\n');
load('./data/detection/train_feats.mat');

% Load train
fprintf('Generating features vectors..\n');
Xtrain = generateFeatureVectors(feats);
ytrain = labels;

% Load test
load('./data/detection/test_feats.mat');
fprintf('Generating test features vectors..\n');
Xtest = generateFeatureVectors(feats);

% Do exp transform
XtrainExp = exp(Xtrain);
XtestExp = exp(Xtest);

% Normalize regular features
fprintf('Normalizing features..\n');
[Xtrain, mu, sigma] = zscore(Xtrain); % train, get mu and std
Xtest = normalize(Xtest, mu, sigma); % normalize test data

% Normalize Exp transform features
fprintf('Normalizing features..\n');
[XtrainExp, muExp, sigmaExp] = zscore(XtrainExp); % train, get mu and std
XtestExp = normalize(XtestExp, muExp, sigmaExp); % normalize test data

%% Apply PCA

PCA.kPC = 100; % Number of PC to keep

fprintf('Performing Principal Component Analysis..\n');
[PCA.coeff, PCA.mu, PCA.latent] = pcaCompute(XtrainExp);
fprintf('PCA > Projecting train and test data on the first %d PC..\n', PCA.kPC);
[pcaXtrain, ~, ~] = pcaApplyOnData(XtrainExp, PCA.coeff, PCA.mu, PCA.kPC);
[pcaXtest, ~, ~] = pcaApplyOnData(XtestExp, PCA.coeff, PCA.mu, PCA.kPC);

fprintf('PCA > Normalizing PCA features..\n');
[pcaXtrain, muPCA, sigmaPCA] = zscore(pcaXtrain);
pcaXtest = normalize(pcaXtest, muPCA, sigmaPCA);

%% Validating model using 2 fold CV (simulate ratio between train / test)

plot_flag = 1;
kfold = 2;
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);

% SVM
learnSVM = @(y, X) trainSVM(y, X, '-t 2 -b 1 -e 0.01');
predictSVM = @(model, X) predictSVM(model, X);

[trAvgTPR_SVM, teAvgTPR_SVM, predTr_SVM, predTe_SVM, trueTr_SVM, trueTe_SVM] = kFoldCrossValidation(ytrain, pcaXtrain, kfold, learnSVM, predictSVM, computePerformance, 1, 'SVM RBF kernel');
savePlot('./report/figures/detection/selected-classifier-validation.pdf','False Positive Rate','True Positive Rate');

%% NN

learnNN = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, 1e-3, [size(X,2) 100 2]);
predictNN = @(model, X) predictNeuralNetwork(model, X);
[trAvgTPR_NN, teAvgTPR_NN, predTr_NN, predTe_NN, trueTr_NN, trueTe_NN] = kFoldCrossValidation(ytrain, XtrainExp, kfold, learnNN, predictNN, computePerformance, plot_flag, 'Neural Network');


%% Train SVM

model = trainSVM(ytrain, pcaXtrain, '-t 2 -b 1 -e 0.01');
Ytest_score = predictSVM(model, pcaXtest);

%% Train NN to check if we have close results

modelNN = trainNeuralNetwork(ytrain, Xtrain, 0, 1, 'sigm', 0, 1e-3, [size(Xtrain,2) 100 2]);
Ytest_scoreNN = predictNeuralNetwork(modelNN, Xtest);

%% Save scores
saveDetectionPredictions(Ytest_score);
