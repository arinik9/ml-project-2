%% Weak prediction on the song recommendation dataset
% Weak prediction: for an existing user, predict unobserved listening counts
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));
clearvars;

%% Dataset pre-processing
% TODO: make a common preprocessing script

% Cell array holding the errors made with various methods
e = {};
e.tr = {}; e.te = {};

% Load dataset
load('./data/recommendation/songTrain.mat');

% Listening counts matrix:
% Each (i, j) corresponds to the listening count of user i for artist j
Yoriginal = Ytrain;
Goriginal = Gtrain;
clear artistName;

%% Outliers removal
% TODO: test removing more or less "outliers"
nDev = 3;
Y = removeOutliersSparse(Yoriginal, nDev);
clearvars nDev;

%% Train / test split
% TODO: cross-validate all the things!
setSeed(1);
% TODO: vary test / train proportions
[~, Ytest, ~, Ytrain, Gtrain] = splitData(Y, Goriginal, 0, 0.1);
[idx, sz] = getRelevantIndices(Ytrain, Ytest);
[testIdx, testSz] = getRelevantIndices(Ytest);

[userDV, artistDV] = generateDerivedVariables(Ytrain);

%% Baseline: constant predictor (overall mean of all observed counts)
[Ynorm, newMean] = normalizedSparse(Ytrain);
YtestNorm = normalizedSparse(Ytest);
% Predict counts (only those for which we have reference data, to save memory)
trYhat0 = sparse(idx.tr.u, idx.tr.a, newMean, sz.tr.u, sz.tr.a);
teYhat0 = sparse(idx.te.u, idx.te.a, newMean, sz.te.u, sz.te.a);

% Compute train and test errors (prediction vs counts in test and training set)
e.tr.constant = computeRmse(Ynorm, trYhat0);
e.te.constant = computeRmse(YtestNorm, teYhat0);

fprintf('RMSE with a constant predictor: %f | %f\n', e.tr.constant, e.te.constant);

% Cleanup
clear overallMean trYhat0 teYhat0;

%% Simple model: predict the average listening count of the artist

% Predict counts based on the artist's average listening count
% (which is one of the derived variables)
trPrediction = zeros(sz.tr.nnz, 1);
tePrediction = zeros(sz.te.nnz, 1);
for k = 1:sz.tr.unique.a
    artist = idx.tr.unique.a(k);
    trPrediction(idx.tr.a == artist) = artistDV(artist, 1);
    tePrediction(idx.te.a == artist) = artistDV(artist, 1);
end;
trYhatMean = sparse(idx.tr.u, idx.tr.a, trPrediction, sz.tr.u, sz.tr.a);
teYhatMean = sparse(idx.te.u, idx.te.a, tePrediction, sz.te.u, sz.te.a);

% Compute train and test errors (prediction vs counts in test and training set)
e.tr.mean = computeRmse(Ytrain, trYhatMean);
e.te.mean = computeRmse(Ytest, teYhatMean);

fprintf('RMSE with a constant predictor per artist: %f | %f\n', e.tr.mean, e.te.mean);

% Cleanup
clearvars i k nCountsObserved meanPerArtist trPrediction tePrediction;
clearvars trYhatMean teYhatMean;

%% ALS-WR
% TODO: experiment different lambdas and number of features
% TODO: cross-validate
nFeatures = 50; % Target reduced dimensionality
lambda = 0.05;
displayLearningCurve = 1;

[U, M] = alswr(Ytrain, Ytest, nFeatures, lambda, displayLearningCurve);

e.tr.als = computeRmse(Ytrain, denormalize(reconstructFromLowRank(U, M, idx, sz), idx));
e.te.als = computeRmse(Ytest, denormalize(reconstructFromLowRank(U, M, testIdx, testSz), testIdx));

fprintf('RMSE ALS-WR (low rank): %f | %f\n', e.tr.als, e.te.als);

% Cleanup
clearvars nFeatures lambda displayLearningCurve;

%% "Each Artist" predictions
% Train a model for each item using derived variables
betas = learnEachArtist(Ytrain, Gtrain, userDV, artistDV);
%%
% Generage predictions
trPrediction = zeros(sz.tr.nnz, 1);
tePrediction = zeros(sz.te.nnz, 1);
for j = 1:sz.tr.unique.a
    artist = idx.tr.unique.a(j);
    
    tXtrain = generateFeatures(artist, Ytrain, Gtrain, userDV, artistDV);
    tXtest = generateFeatures(artist, Ytest, Gtrain, userDV, artistDV);
   
    % Since Xtrain has exactly as many lines as users listened to this
    % artist, this will automatically produce only the predictions we need
    trPrediction(idx.tr.a == artist) = tXtrain * betas(:, artist);
    tePrediction(idx.te.a == artist) = tXtest * betas(:, artist);
end;

trYhatLS = sparse(idx.tr.u, idx.tr.a, trPrediction, sz.tr.u, sz.tr.a);
teYhatLS = sparse(idx.te.u, idx.te.a, tePrediction, sz.te.u, sz.te.a);

e.tr.leastSquares = computeRmse(Ytrain, denormalize(trYhatLS, idx));
e.te.leastSquares = computeRmse(Ytest, denormalize(teYhatLS, idx));

fprintf('RMSE with Least-Squares on head only : %f | %f\n', e.tr.leastSquares, e.te.leastSquares);
diagnoseError(Ytrain, trYhatLS);

% Cleanup
clearvars j artist beta Xtrain tXtrain Xtest tXtest trPrediction tePrediction trVal teVal;
clearvars yTrain yTest trYhatLS teYhatLS;

%% Other predictions
% TODO
