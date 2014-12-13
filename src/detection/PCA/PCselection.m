getStartedDetection;

%%

[PCA.explained, PCA.cumExplained] = pcaExplainedVariance(PCA.latent, 1, 1000);
savePlot('./report/figures/detection/pca-varianceExplained.pdf','Principal Components','Variance Explained');

%% Different projections

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 50);
[pcaX50, ~, ~] = pcaApplyOnData(X, PCA.coeff, PCA.mu, 50);
[pcaX50, ~, ~] = zscore(pcaX50);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 75);
[pcaX75, ~, ~] = pcaApplyOnData(X, PCA.coeff, PCA.mu, 50);
[pcaX75, ~, ~] = zscore(pcaX75);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 100);
[pcaX100, ~, ~] = pcaApplyOnData(X, PCA.coeff, PCA.mu, 100);
[pcaX100, ~, ~] = zscore(pcaX100);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 300);
[pcaX300, ~, ~] = pcaApplyOnData(X, PCA.coeff, PCA.mu, 300);
[pcaX300, ~, ~] = zscore(pcaX300);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 500);
[pcaX500, ~, ~] = pcaApplyOnData(X, PCA.coeff, PCA.mu, 500);
[pcaX500, ~, ~] = zscore(pcaX500);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 1000);
[pcaX1000, ~, ~] = pcaApplyOnData(X, PCA.coeff, PCA.mu, 1000);
[pcaX1000, ~, ~] = zscore(pcaX1000);

%% Train and predict Logistic Regression model

plot_flag = 0;
learn = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, 0, [size(X,2),2]);
predict = @(model, X) predictNeuralNetwork(model, X);
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 0, 0, model_name);

[trAvgTPR_50, teAvgTPR_50, predTr_50, predTe_50, trueTr_50, trueTe_50] = kFoldCrossValidation(y, pcaX50, 3, learn, predict, computePerformance, plot_flag, 'Logistic Regression');

[trAvgTPR_75, teAvgTPR_75, predTr_75, predTe_75, trueTr_75, trueTe_75] = kFoldCrossValidation(y, pcaX75, 3, learn, predict, computePerformance, plot_flag, 'Logistic Regression');

[trAvgTPR_100, teAvgTPR_100, predTr_100, predTe_100, trueTr_100, trueTe_100] = kFoldCrossValidation(y, pcaX100, 3, learn, predict, computePerformance, plot_flag, 'Logistic Regression');

[trAvgTPR_300, teAvgTPR_300, predTr_300, predTe_300, trueTr_300, trueTe_300] = kFoldCrossValidation(y, pcaX300, 3, learn, predict, computePerformance, plot_flag, 'Logistic Regression');

[trAvgTPR_500, teAvgTPR_500, predTr_500, predTe_500, trueTr_500, trueTe_500] = kFoldCrossValidation(y, pcaX500, 3, learn, predict, computePerformance, plot_flag, 'Logistic Regression');

[trAvgTPR_1000, teAvgTPR_1000, predTr_1000, predTe_1000, trueTr_1000, trueTe_1000] = kFoldCrossValidation(y, pcaX1000, 3, learn, predict, computePerformance, plot_flag, 'Logistic Regression');

%% Evaluate multiple results

predictRandom = rand(size(trueTe_1000)); 

% Methods names for legend
methodNames = {'50PC', '75PC', '100PC', '300PC', '500PC', '1000PC'};

% Prediction performances on different models
avgTPRList = kCVevaluateMultipleMethods( cat(3, trueTe_50, trueTe_75, trueTe_100, trueTe_300, trueTe_500, trueTe_1000), cat(3, predTe_50, predTe_75, predTe_100, predTe_300, predTe_500, predTe_1000), true, methodNames );

