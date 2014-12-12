clearvars;

% add path to source files and toolboxs
addpath(genpath('./toolbox/'));
addpath(genpath('./src/'));

% Load both features and training images
load('./data/detection/train_feats.mat');
% load('./data/detection/train_imgs.mat');

%% Pre-process data

fprintf('Generating feature vectors..\n');
X = generateFeatureVectors(feats);
y = labels;

% Split data into train and test set given a proportion
prop = 2/3;
fprintf('Splitting into train/test with proportion %.2f..\n', prop);
[Tr.X, Tr.y, Te.X, Te.y] = splitDataDetection(y, X, prop);

% Normalize
fprintf('Normalizing features..\n');
[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

% if not using Piotr's toolbox (to replace normalize function)
% onesX = ones(size(Te.X,1), 1);
% Te.normX = (Te.X - onesX * mu) ./ (onesX * sigma); 


fprintf('Done! You can start playing with the features!\n');

%% Principal Component Analysis
fprintf('Performing Principal Component Analysis..\n');

[PCA.coeff, PCA.mu, PCA.latent] = pcaCompute(Tr.normX);


%% Plot PCA Representation
fprintf('Computing percentage of the total variance explained by each principal component..\n');
% Might be useful to choose how many PC we are keeping.
% On lower dimension dataset we usually keep PC so that we have a
% cumulative percentage of 95% of the total variance but here it would be
% too many PC.

[PCA.explained, PCA.cumExplained] = pcaExplainedVariance(PCA.latent, 0);

%% Apply PCA to train and test data
% Project train and test data in the space formed by the PC we have decided
% to keep. The features obtained are not normalized so we do normalize them 
% at the end

fprintf('Projecting train and test data into the lower space..\n');

% Number of PC kept (to choose)
PCA.kPC = 50;

if exist('PCA.cumExplained','var')
   fprintf('We are projecting on the first %d principal components (%.2f%% explained variance)..\n', PCA.kPC, PCA.cumExplained(PCA.kPC)*100);
else 
   fprintf('We are projecting on the first %d principal components..\n', PCA.kPC);
end

[Tr.pcaX, Tr.pcaXhat, Tr.pcaAvsq] = pcaApplyOnData(Tr.normX, PCA.coeff, PCA.mu, PCA.kPC);
[Te.pcaX, Te.pcaXhat, Te.pcaAvsq] = pcaApplyOnData(Te.normX, PCA.coeff, PCA.mu, PCA.kPC);

% Normalize reduced input features
[Tr.pcaX, mu, sigma] = zscore(Tr.pcaX);
Te.pcaX = normalize(Te.pcaX, mu, sigma);

fprintf('Done! We have now reduced train and test set !\n');

%% Random forest
fprintf('Using random forest...\n');

% Number of trees in the forest
nTrees = 50;

fprintf('Training random forest with %d trees...\n', nTrees);
% Train random forest
B = TreeBagger(nTrees, Tr.X, Tr.y); % default implementation is for classification

fprintf('Predicting on the test set...\n', nTrees);
% Make predictions using the trained random forest
[labels, scores] = B.predict(Te.X);

% Predictions is a char though. We want it to be a number.
rfPred = scores(:,1) - scores(:,2);



%% Logistic Regression
% Logistic Regression implementation using NN: A simple layer NN is a logistic regression
% We use values from the PCA

logRegPred = neuralNetworkPredict(Tr.y, Tr.pcaX, Te.pcaX, 0, 1, 'sigm', 0, 0, [size(Tr.pcaX,2) 2]);
%logRegPredTrain = neuralNetworkPredict(Tr.y, Tr.pcaX, Tr.pcaX, 0, 1, 'sigm', 0, 0, [size(Tr.pcaX,2) 2]);


%% Prediction with different NN
fprintf('Predictions using different Neural Networks..\n');

fprintf('Default NN prediction\n');
% Default NN prediction (tanh, learningRate=1, ...)
%nnPred = neuralNetworkPredict(Tr, Te);

fprintf('Tuned NN prediction\n');
% Tuned NN prediction (sigmoid activations and lower learningRate)
%nnPred = neuralNetworkPredict(Tr, Te, 0, 1, 'sigm');

% Tuned NN prediction with Dropout Fraction set to 0.5 (close to optimality
% according to paper on dropout)
% nnPred3 = neuralNetworkPredict(Tr.y, Tr.normX, Te.normX, 0, 1, 'sigm', 0, 1e-4);
nn = learnNeuralNetwork(Tr.y, Tr.normX, 0, 1, 'sigm', 0, 1e-4);
nnPred = predictNeuralNetwork(nn, Te.normX);


% Tunned NN predictions with Weight Decay on L2 (Tikhonov regularization)
%nnPred3 = neuralNetworkPredict(Tr, Te, 0, 1, 'sigm', 0, 1e-4);

%% GP large scale Classification

x = Tr.pcaNormX;
y = Tr.y;
t = Te.pcaNormX;
nPlot = size(t,1);

tic;
% gpPred = GPClassificationPrediction(y,x,t);
gpModel = learnGPClassification(y, x);
gpPred = predictGPClassification(gpModel, t);
TimeSpent = toc;

yHatGP = outputLabelsFromPrediction(gpPred, 0.5);

% We do recover our large rate of negative VS positive images
hist(yHatGP);

%% Random Predictions

fprintf('Random prediction\n');
% Random prediction
randPred = rand(size(Te.y)); 
randPredTrain = rand(size(Tr.y)); 

%% Test One Model
% Methods names for legend
methodNames = {'Model','Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( Te.y > 0, [logRegPred, randPred], true, methodNames );
%avgTPRListTr = evaluateMultipleMethods( Tr.y > 0, [nnPred3, randPredTrain], true, methodNames );

%%
% Methods names for legend
methodNames = {'0.4','0.5', '0.6', '0.7', '0.8', '0.9', '0.95'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( Te.y > 0, [yHatRandom, yHatRandom2, yHatRandom3, yHatRandom4, yHatRandom5, yHatRandom6, yHatRandom7], true, methodNames );
