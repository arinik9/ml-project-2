%% Weak prediction on the song recommendation dataset
% Weak prediction: for an existing user, predict unobserved listening counts
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

%% Dataset pre-processing
% TODO: refactor

clearvars;

% Load dataset
load('./data/recommendation/songTrain.mat');

% Counts matrix:
% Each (i, j) corresponds to the listening count of user i for artist j
Yoriginal = Ytrain;
Goriginal = Gtrain;

%% Train / test split
% TODO: cross-validate all the things!
% TODO: no need to withold users for strong prediction if we're focusing on
% weak prediction here
[~, Ytest, ~, Ytrain, ~] = splitData(Yoriginal, Goriginal);
% Total size of train and test matrices
[trN, trD] = size(Ytrain);
[teN, teD] = size(Ytest);
% Indices of available counts
[trUserIndices, trArtistIndices] = find(Ytrain);
[teUserIndices, teArtistIndices] = find(Ytest);

%% Baseline: constant predictor (overall mean of all observed counts)
overallMean = mean(nonzeros(Ytrain));

% Predict counts (only those we're going to test on, to save memory)
% TODO: should we predict *all* counts?
values = overallMean * nnz(Ytrain);
trYhat0 = sparse(trUserIndices, trArtistIndices, values, trN, trD);
teYhat0 = sparse(teUserIndices, teArtistIndices, values, teN, teD);

% Compute train and test errors (prediction vs counts in test and training set)
trErr0 = computeRmse(Ytrain, trYhat0);
teErr0 = computeRmse(Ytest, teYhat0);

fprintf('RMSE with a constant predictor: %f | %f\n', trErr0, teErr0);
