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
setSeed(1);
% TODO: vary test / train proportions
[~, Ytest, ~, Ytrain, ~] = splitData(Yoriginal, Goriginal, 0, 0.1);
% Total size of train and test matrices
[trN, trD] = size(Ytrain);
[teN, teD] = size(Ytest);
% Indices of available counts (expressed in the same coordinates space)
[trUserIndices, trArtistIndices] = find(Ytrain);
[teUserIndices, teArtistIndices] = find(Ytest);

% Cleanup: we're not using the social graph for weak prediction
% TODO: should we?
clear artistName Goriginal Gtrain;

%% Baseline: constant predictor (overall mean of all observed counts)
overallMean = mean(nonzeros(Ytrain));

% Predict counts (only those for which we have reference data, to save memory)
% TODO: should we predict *all* counts?
trYhat0 = sparse(trUserIndices, trArtistIndices, overallMean, trN, trD);
teYhat0 = sparse(teUserIndices, teArtistIndices, overallMean, teN, teD);

% Compute train and test errors (prediction vs counts in test and training set)
trErr0 = computeRmse(Ytrain, trYhat0);
teErr0 = computeRmse(Ytest, teYhat0);

fprintf('RMSE with a constant predictor: %f | %f\n', trErr0, teErr0);

% Cleanup
clear overallMean;

%% Simple model: predict the average listening count of the user

% Compute corresponding mean value for each user
uniqueUsers = unique(trUserIndices);
meanPerUser = zeros(trN, 1);
for i = 1:length(uniqueUsers)
    nCountsObserved = nnz(Ytrain(uniqueUsers(i), :));
    meanPerUser(i) = sum(Ytrain(uniqueUsers(i), :), 2) / nCountsObserved;
end

% Predict counts (only those for which we have reference data, to save memory)
trPrediction = zeros(length(trUserIndices), 1);
for k = 1:length(trPrediction)
    trPrediction(k) = meanPerUser(trUserIndices(k));
end;
tePrediction = zeros(length(teUserIndices), 1);
for k = 1:length(tePrediction)
    tePrediction(k) = meanPerUser(teUserIndices(k));
end;

trYhatMean = sparse(trUserIndices, trArtistIndices, trPrediction, trN, trD);
teYhatMean = sparse(teUserIndices, teArtistIndices, tePrediction, teN, teD);

% Compute train and test errors (prediction vs counts in test and training set)
trErrMean = computeRmse(Ytrain, trYhatMean);
teErrMean = computeRmse(Ytest, teYhatMean);

fprintf('RMSE with a constant predictor per user: %f | %f\n', trErrMean, teErrMean);

% Cleanup
clearvars i k nCountsObserved uniqueUsers meanPerUser trPrediction tePrediction;

%% Other predictions
% TODO
