%% Weak prediction on the song recommendation dataset
% Weak prediction: for an existing user, predict unobserved listening counts
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

%% Dataset pre-processing
% TODO: make a common preprocessing script

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

% Cleanup
clear artistName Goriginal Gtrain;

%% Outliers removal & normalization
% TODO: test removing more or less "outliers"
nDev = 3;
[Ytrain, Ytest] = removeOutliers(Ytrain, nDev, Ytest);

% TODO: denormalize after prediction to obtain the correct scale
[Ytrain, ~, ~, Ytest] = normalizedByUsers(Ytrain, Ytest);

% Total size of train and test matrices
[trN, trD] = size(Ytrain);
[teN, teD] = size(Ytest);
% Indices of available counts (expressed in the same coordinates space)
[trUserIndices, trArtistIndices] = find(Ytrain);
[teUserIndices, teArtistIndices] = find(Ytest);

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

%% Simple model: predict the average listening count of the artist

% Compute corresponding mean value for each artist
uniqueArtists = unique(trArtistIndices);
meanPerArtist = zeros(trD, 1);
for i = 1:length(uniqueArtists)
    nCountsObserved = nnz(Ytrain(:, uniqueArtists(i)));
    if(nCountsObserved > 0)
        meanPerArtist(i) = sum(Ytrain(:, uniqueArtists(i))) / nCountsObserved;
    else
        meanPerArtist(i) = 0;
    end;
end

% Predict counts (only those for which we have reference data, to save memory)
trPrediction = zeros(length(trArtistIndices), 1);
for k = 1:length(trPrediction)
    trPrediction(k) = meanPerArtist(trArtistIndices(k));
end;
tePrediction = zeros(length(teArtistIndices), 1);
for k = 1:length(tePrediction)
    tePrediction(k) = meanPerArtist(teArtistIndices(k));
end;

trYhatMean = sparse(trUserIndices, trArtistIndices, trPrediction, trN, trD);
teYhatMean = sparse(teUserIndices, teArtistIndices, tePrediction, teN, teD);

% Compute train and test errors (prediction vs counts in test and training set)
trErrMean = computeRmse(Ytrain, trYhatMean);
teErrMean = computeRmse(Ytest, teYhatMean);

fprintf('RMSE with a constant predictor per artist: %f | %f\n', trErrMean, teErrMean);

% Cleanup
clearvars i k nCountsObserved uniqueUsers meanPerUser trPrediction tePrediction;

%% Other predictions
% TODO
