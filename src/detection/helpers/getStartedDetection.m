clearvars;

% Add path and load data ---------------------------
addpath(genpath('./toolbox/'));
addpath(genpath('./src/'));

fprintf('Load data..\n');
load('./data/detection/train_feats.mat');

% Generate features ---------------------------------
fprintf('Generating features vectors..\n');
X = generateFeatureVectors(feats);
y = labels;

% Normalize -----------------------------------------
fprintf('Normalizing features..\n');
[X, ~, ~] = zscore(X);

% Principal Component Analysis ----------------------
fprintf('Performing Principal Component Analysis..\n');
[PCA.coeff, PCA.mu, PCA.latent] = pcaCompute(X);

PCA.kPC = 100; % Number of PC kept (to choose). Chosen from the PCAselection study

fprintf('PCA > Projecting train and test data on the first %d PC..\n', PCA.kPC);
[pcaX, pcaXhat, pcaAvsq] = pcaApplyOnData(X, PCA.coeff, PCA.mu, PCA.kPC);

% Normalize PCA features ----------------------------
fprintf('PCA > Normalizing PCA features..\n');
[pcaX, ~, ~] = zscore(pcaX);

% ---------------------------------------------------
% Working with the exp() transform
% ---------------------------------------------------

% Feature transformations ---------------------------
fprintf('Feature tranformation: exp(X)..\n');
expX = exp(X);

% Normalize -----------------------------------------
fprintf('Normalizing features..\n');
[expX, ~, ~] = zscore(expX); % train, get mu and std

% Principal Component Analysis ----------------------
fprintf('Performing Principal Component Analysis on Exp(X)..\n');
[PCAexpX.coeff, PCAexpX.mu, PCAexpX.latent] = pcaCompute(expX);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', PCA.kPC);
[pcaExpX, ~, ~] = pcaApplyOnData(expX, PCAexpX.coeff, PCAexpX.mu, PCA.kPC);

% Normalize PCA features ----------------------------
fprintf('PCA > Normalizing PCA features..\n');
[pcaExpX, ~, ~] = zscore(pcaExpX);
