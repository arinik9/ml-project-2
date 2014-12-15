%% Weak prediction on the song recommendation dataset
% Weak prediction: for an existing user, predict unobserved listening counts
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));
clearvars;
%%
loadDataset;
% Number of random train / test splits to generate
nSplits = 5;

% Shortcut
evaluate = @(name, learn) evaluateMethod(name, learn, Yoriginal, Goriginal, nSplits, 1);

%% Baseline: constant predictor (overall mean of all observed counts)
name = 'Constant';
[e.tr.(name), e.te.(name)] = evaluate(name, @learnConstantPredictor);

%% Simple model: predict the average listening count of the user
% (which is one of the derived variables)
name = 'UserMean';
[e.tr.(name), e.te.(name)] = evaluate(name, @learnAveragePerUserPredictor);

%% ALS-WR (low rank approximation)
% We tried: lambda = 0.01, 0.025, 0.05, 0.075, 0.1, 0.2
% TODO: select hyper-parameters by cross-validation
nFeatures = 250; % Target reduced dimensionality
lambda = 0.5;
displayLearningCurve = 1;
learnAlsWr = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda, displayLearningCurve);

name = 'ALSWR';
[e.tr.(name), e.te.(name), trLambda7, teLambda7] = evaluate(name, learnAlsWr);

%% "Each Artist" predictions
% Train a separate model for each artist using derived variables
% We clearly see that it overfits badly when too few counts are available.
name = 'EachArtist';
[e.tr.(name), e.te.(name)] = evaluate(name, @learnEachArtistPredictor);

%% Head / tail predictor
% Train a separate model for each artist of the head
% Train a common model for each cluster of tail artists
% TODO: select threshold with cross-validation?
headThreshold = 10;
learnHeadTail = @(Y, Ytest, userDV, artistDV) learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold);

name = ['HeadTail', int2str(headThreshold)];
[e.tr.(name), e.te.(name)] = evaluate(name, learnHeadTail);

%% K-Means clustering
% Maximum number of clusters (may not all be used)
% TODO: select K with cross-validation
K = 500;
% Parameters for ALS-WR
% Our goal here is to obtain a version of Ytrain but with lower
% dimensionality. We're not trying to predict from the result, so we
% can overfit completely.
%nFeatures = 20;
%lambda = 0.000001;

%reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 0)';
%getSimilarity = @(Ytrain, Ytest, userDV) computeSimilarityMatrix(Ytrain, Ytest, userDV, reduceSpace);

%learnKMeansALS = @(Y, Ytest, userDV, artistDV) ...
%    learnKMeansPredictor(Y, Ytest, userDV, artistDV, K, getSimilarity(Y, Ytest, userDV));

learnKMeans = @(Y, Ytest, userDV, artistDV) ...
    learnKMeansPredictor(Y, Ytest, userDV, artistDV, K);

name = ['K', int2str(K), 'Means'];
[e.tr.(name), e.te.(name)] = evaluate(name, learnKMeans);

clearvars K nFeatures lambda;

%% Gaussian Mixture Model clustering (soft clustering)
K = 15;
nFeatures = 40;
lambda = 0.05;

reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 1);
learnGMM = @(Y, Ytest, userDV, artistDV) ...
    learnGMMPredictor(Y, Ytest, userDV, artistDV, reduceSpace, K);

name = ['GMM', int2str(K), 'ALS'];
[e.tr.(name), e.te.(name)] = evaluate(name, learnGMM);

clearvars K nFeatures lambda;

%% Top-K recommendation (on the full dataset using dim-reduction)
% TODO: select K with cross-validation
K = 150;
nFeatures = 200;
lambda = 0.000001;

reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 0)';
getSimilarity = @(Ytrain, Ytest, userDV) computeSimilarityMatrix(Ytrain, Ytest, userDV, reduceSpace);
learnTopKALS = @(Y, Ytest, userDV, artistDV) ...
    learnTopKPredictor(Y, Ytest, userDV, artistDV, K, getSimilarity(Y, Ytest, userDV));

name = ['Top', int2str(K), 'NeighborsALS'];
[e.tr.(name), e.te.(name)] = evaluate(name, learnTopKALS);

clearvars K nFeatures lambda;


%% Similarity-based predictor
% Can be seen as a particular case of the Top-K recommendation,
% where K is equal to the total number of individuals.
name = 'SimilarityBased';
%transform = @(S) S;
%transform = @applyFisherTransform;
learnSimilarity = @(Y, Ytest, userDV, artistDV) ...
    learnSimilarityBasedPredictor(Y, Ytest, userDV, artistDV);

[e.tr.(name), e.te.(name), trErrorSim, teErrorSim] = evaluate(name, learnSimilarity);

