clearvars;

% add path to source files and toolboxs
addpath(genpath('./toolbox/'));
addpath(genpath('./src/'));

% Load both features and training images
load('./data/detection/train_feats.mat');
% load('./data/detection/train_imgs.mat');

%% Pre-processing data

fprintf('Generating feature vectors..\n');
X = generateFeatureVectors(feats);
y = labels;

% Normalize
fprintf('Normalizing features..\n');
[X, ~, ~] = zscore(X); % train, get mu and std

fprintf('Done! Data are ready to play with..\n');

%% Principal Component Analysis (to refactor?)
fprintf('Performing Principal Component Analysis..\n');

% Compute principal components, mean and eigen values
[PCA.coeff, PCA.mu, PCA.latent] = pca(X');

% Percentage of the total variance explained by each principal component
PCA.explained = PCA.latent ./sum(PCA.latent);
% Cumulative percentage of the total variance explained by each principal component
PCA.explainedCum = (cumsum(PCA.latent)./sum(PCA.latent));

% Number of PC kept (to choose)
PCA.kPC = 50;
fprintf('We are projecting on the first %d principal components..\n', PCA.kPC);

fprintf('Projecting train and test data into the lower space..\n');
[pcompX, pcompXhat, PCA.avsq] = pcaApply(X', PCA.coeff, PCA.mu, PCA.kPC);
pcaX = pcompX'; pcaXhat = pcompXhat';

% Normalize reduced input features
[pcaX, ~, ~] = zscore(pcaX);

fprintf('Done! We have now reduced train and test set !\n');

clear pcompX pcompXhat;

%% kCV

learn = @(y, X) learnNeuralNetwork(y, X, 0, 1, 'sigm', 0, 0, [size(X,2) 2]);
predict = @(model, X) predictNeuralNetwork(model, X);
[trAvgTPR, teAvgTPR, predTr, predTe] = kFoldCrossValidation(y, pcaX, 3, learn, predict, 0, 'Logistic Regression');

%TODO: kCV on other models
%% Learn parameters using kCV

dpFractions = [0 0.1 0.2 0.3 0.4 0.45 0.5 0.55 0.6];
wDecays = [1e-5 1e-4 1e-3 1e-2 1e-1];
[bestDp, bestWd trainTPR, testTPR] = findHyperParametersNeuralNetwork(y, pcaX, 3, dpFractions, wDecays, 1);

% bestDp: 0.2000 bestWd: 1.0000e-04

%% Test results


% Methods names for legend
methodNames = {'Tuned NN','Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( Te.y > 0, [rfPred, randPred], true, methodNames );