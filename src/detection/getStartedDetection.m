clearvars;

% add path to source files and toolboxs -------------
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
[X, ~, ~] = zscore(X); % train, get mu and std

% Principal Component Analysis ----------------------
fprintf('Performing Principal Component Analysis..\n');
[PCA.coeff, PCA.mu, PCA.latent] = pcaCompute(X);

PCA.kPC = 50; % Number of PC kept (to choose)

fprintf('PCA > Projecting train and test data on the first %d PC..\n', PCA.kPC);
[pcaX, pcaXhat, pcaAvsq] = pcaApplyOnData(X, PCA.coeff, PCA.mu, PCA.kPC);

% Normalize PCA features
fprintf('PCA > Normalizing PCA features..\n');
[pcaX, ~, ~] = zscore(pcaX);


% Feature transformations
fprintf('Feature tranformation: exp(X)..\n');
expX = exp(X);

% Normalize -----------------------------------------
fprintf('Normalizing features..\n');
[expX, ~, ~] = zscore(expX); % train, get mu and std

fprintf('Performing Principal Component Analysis on Exp(X)..\n');
[PCAexpX.coeff, PCAexpX.mu, PCAexpX.latent] = pcaCompute(expX);

fprintf('PCA > Projecting train and test data on the first %d PC..\n', PCA.kPC);
[pcaExpX, ~, ~] = pcaApplyOnData(expX, PCAexpX.coeff, PCAexpX.mu, PCA.kPC);

% Normalize PCA features
fprintf('PCA > Normalizing PCA features..\n');
[pcaExpX, ~, ~] = zscore(pcaExpX);
%}