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

%% Outliers removal & normalization
% TODO: test removing more or less "outliers"
nDev = 3;
Y = removeOutliersSparse(Yoriginal, nDev);

% TODO: denormalize after prediction to obtain the correct scale
% TODO: we're normalizing before the train/test split, is this correct?
Y = normalizedByUsers(Y);

clearvars nDev;

%% Train / test split
% TODO: cross-validate all the things!
setSeed(1);
% TODO: vary test / train proportions
[~, Ytest, ~, Ytrain, Gtrain] = splitData(Y, Goriginal, 0, 0.1);

[idx, sz] = getRelevantIndices(Ytrain, Ytest);
[userDV, artistDV] = generateDerivedVariables(Ytrain);

%% Baseline: constant predictor (overall mean of all observed counts)
overallMean = mean(nonzeros(Ytrain));

% Predict counts (only those for which we have reference data, to save memory)
% TODO: should we predict *all* counts?
trYhat0 = sparse(idx.tr.u, idx.tr.a, overallMean, sz.tr.u, sz.tr.a);
teYhat0 = sparse(idx.te.u, idx.te.a, overallMean, sz.te.u, sz.te.a);

% Compute train and test errors (prediction vs counts in test and training set)
e.tr.constant = computeRmse(Ytrain, trYhat0);
e.te.constant = computeRmse(Ytest, teYhat0);

fprintf('RMSE with a constant predictor: %f | %f\n', e.tr.constant, e.te.constant);

% Cleanup
clear overallMean trYhat0 teYhat0;

%% Simple model: predict the average listening count of the artist

% Compute corresponding mean value for each artist
meanPerArtist = zeros(sz.tr.a, 1);
for j = 1:sz.tr.unique.a
    artist = idx.tr.unique.a(j);
    nCountsObserved = nnz(Ytrain(:, artist));
    if(nCountsObserved > 0)
        meanPerArtist(j) = sum(Ytrain(:, artist)) / nCountsObserved;
    else
        meanPerArtist(j) = 0;
    end;
end

% Predict counts (only those for which we have reference data, to save memory)
trPrediction = zeros(sz.tr.nnz, 1);
for k = 1:sz.tr.nnz
    trPrediction(k) = meanPerArtist(idx.tr.a(k));
end;
tePrediction = zeros(sz.te.nnz, 1);
for k = 1:sz.te.nnz
    tePrediction(k) = meanPerArtist(idx.te.a(k));
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

e.tr.als = computeRmse(Ytrain, reconstructFromLowRank(Ytrain, U, M));
e.te.als = computeRmse(Ytest, reconstructFromLowRank(Ytest, U, M));

fprintf('RMSE ALS-WR (low rank): %f | %f\n', e.tr.als, e.te.als);

% Cleanup
clearvars nFeatures lambda displayLearningCurve U M;

%% "Each Artist" predictions
% Train a model for each item using derived variables

[userDV, artistDV] = generateDerivedVariables(Ytrain);

% TODO: refactor

% Head / tail split
headThreshold = 100;

% Error made for each artist
leastSquaresErrors = zeros(1, length(uniqueArtists));

trValues = zeros(length(trArtistIndices), 1);
teValues = zeros(length(teArtistIndices), 1);
for j = 1:length(uniqueArtists)
    artist = uniqueArtists(j);
    
    yTrain = nonzeros(Ytrain(:, artist));
    yTest = nonzeros(Ytest(:, artist));
    
    if(nnz(Ytrain(:, artist)) > headThreshold)
        % Train a linear model for artist j
        Xtrain = generateFeatures(artist, Ytrain, Gtrain, userDV, artistDV);
        tXtrain = [Xtrain ones(size(Xtrain, 1), 1)];
        Xtest = generateFeatures(artist, Ytest, Gtrain, userDV, artistDV);
        tXtest = [Xtest ones(size(Xtest, 1), 1)];
        
        % Simple least squares
        beta = (tXtrain' * tXtrain) \ (tXtrain' * yTrain);
        trVal = tXtrain * beta;
        teVal = tXtest * beta;
    else
        trVal = 0;
        teVal = 0;
    end;
    
    trValues(trArtistIndices == artist) = trVal;
    teValues(teArtistIndices == artist) = teVal;
    
    leastSquaresErrors(j) = computeRmse(yTest, teVal);
end;

hist(nonzeros(leastSquaresErrors));

trYhatLS = sparse(trUserIndices, trArtistIndices, trValues, sz.tr.n, sz.tr.d);
teYhatLS = sparse(teUserIndices, teArtistIndices, teValues, sz.te.n, sz.te.d);

e.tr.leastSquares = computeRmse(Ytrain, trYhatLS);
e.te.leastSquares = computeRmse(Ytest, teYhatLS);

fprintf('RMSE with Least-Squares on head only : %f | %f\n', e.tr.leastSquares, e.te.leastSquares);

% Cleanup
clearvars j artist beta Xtrain tXtrain Xtest tXtest trValues trVal teVal;
clearvars yTrain yTest teValues trYhatLS teYhatLS;

%% Other predictions
% TODO
