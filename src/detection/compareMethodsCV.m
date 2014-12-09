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
[trAvgTPR, teAvgTPR, predTr, predTe] = kFoldCrossValidation(y, pcaX, 2, learn, predict);

