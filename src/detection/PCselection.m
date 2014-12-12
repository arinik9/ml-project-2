getStartedDetection;

%% Split

prop = 2/3;
fprintf('Splitting into train/test with proportion %.2f..\n', prop);
[Tr.X, Tr.y, Te.X, Te.y] = splitDataDetection(y, X, prop);

%% Different projections

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 50);
[Tr.pcaX50, ~, ~] = pcaApplyOnData(Tr.X, PCA.coeff, PCA.mu, 50);
[Te.pcaX50, ~, ~] = pcaApplyOnData(Te.X, PCA.coeff, PCA.mu, 50);
% Normalize PCA features
fprintf('PCA > Normalizing PCA features..\n');
[Tr.pcaX50, mu50, sigma50] = zscore(Tr.pcaX50);
Te.pcaX50 = normalize(Te.pcaX50, mu50, sigma50);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 100);
[Tr.pcaX100, ~, ~] = pcaApplyOnData(Tr.X, PCA.coeff, PCA.mu, 100);
[Te.pcaX100, ~, ~] = pcaApplyOnData(Te.X, PCA.coeff, PCA.mu, 100);
% Normalize PCA features
fprintf('PCA > Normalizing PCA features..\n');
[Tr.pcaX100, mu100, sigma100] = zscore(Tr.pcaX100);
Te.pcaX100 = normalize(Te.pcaX100, mu100, sigma100);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 300);
[Tr.pcaX300, ~, ~] = pcaApplyOnData(Tr.X, PCA.coeff, PCA.mu, 300);
[Te.pcaX300, ~, ~] = pcaApplyOnData(Te.X, PCA.coeff, PCA.mu, 300);
% Normalize PCA features
fprintf('PCA > Normalizing PCA features..\n');
[Tr.pcaX300, mu300, sigma300] = zscore(Tr.pcaX300);
Te.pcaX300 = normalize(Te.pcaX300, mu300, sigma300);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 500);
[Tr.pcaX500, ~, ~] = pcaApplyOnData(Tr.X, PCA.coeff, PCA.mu, 500);
[Te.pcaX500, ~, ~] = pcaApplyOnData(Te.X, PCA.coeff, PCA.mu, 500);
% Normalize PCA features
fprintf('PCA > Normalizing PCA features..\n');
[Tr.pcaX500, mu500, sigma500] = zscore(Tr.pcaX500);
Te.pcaX500 = normalize(Te.pcaX500, mu500, sigma500);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', 1000);
[Tr.pcaX1000, ~, ~] = pcaApplyOnData(Tr.X, PCA.coeff, PCA.mu, 1000);
[Te.pcaX1000, ~, ~] = pcaApplyOnData(Te.X, PCA.coeff, PCA.mu, 1000);
% Normalize PCA features
fprintf('PCA > Normalizing PCA features..\n');
[Tr.pcaX1000, mu1000, sigma1000] = zscore(Tr.pcaX1000);
Te.pcaX1000 = normalize(Te.pcaX1000, mu1000, sigma1000);


%%

model50 = trainNeuralNetwork(Tr.y, Tr.pcaX50, 0, 1, 'sigm', 0, 1.e-4, [size(Tr.pcaX50,2), 2]);
predict50 = predictNeuralNetwork(model50, Te.pcaX50);

model100 = trainNeuralNetwork(Tr.y, Tr.pcaX100, 0, 1, 'sigm', 0, 1.e-4, [size(Tr.pcaX100,2), 2]);
predict100 = predictNeuralNetwork(model100, Te.pcaX100);

model300 = trainNeuralNetwork(Tr.y, Tr.pcaX300, 0, 1, 'sigm', 0, 1.e-4, [size(Tr.pcaX300,2), 2]);
predict300 = predictNeuralNetwork(model300, Te.pcaX300);

model500 = trainNeuralNetwork(Tr.y, Tr.pcaX500, 0, 1, 'sigm', 0, 1.e-4, [size(Tr.pcaX500,2), 2]);
predict500 = predictNeuralNetwork(model500, Te.pcaX500);

model1000 = trainNeuralNetwork(Tr.y, Tr.pcaX1000, 0, 1, 'sigm', 0, 1.e-4, [size(Tr.pcaX1000,2), 2]);
predict1000 = predictNeuralNetwork(model1000, Te.pcaX1000);

%% Evaluate multiple results

predictRandom = rand(size(Te.y)); 

% Methods names for legend
methodNames = {'50PC','100PC', '300PC', '500PC', '1000PC', 'Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( Te.y > 0, [predict50, predict100, predict300, predict500, predict1000, predictRandom], true, methodNames );

