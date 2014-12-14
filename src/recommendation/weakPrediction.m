%% Weak prediction on the song recommendation dataset
% Weak prediction: for an existing user, predict unobserved listening counts
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));
clearvars;
%%
loadDataset;
% Number of random train / test splits to generate
% TODO: moar
nSplits = 2;

% Shortcut
evaluate = @(name, learn) evaluateMethod(name, learn, Yoriginal, Goriginal, nSplits, 1);

%% Baseline: constant predictor (overall mean of all observed counts)
name = 'Constant';
[e.tr.(name), e.te.(name)] = evaluate(name, @learnConstantPredictor);

%% Simple model: predict the average listening count of the artist
% (which is one of the derived variables)
name = 'ArtistMean';
[e.tr.(name), e.te.(name)] = evaluate(name, @learnAveragePerArtistPredictor);

%% ALS-WR
% TODO: experiment different lambdas and number of features
% TODO: select hyper-parameters by cross-validation
nFeatures = 50; % Target reduced dimensionality
lambda = 0.01;
displayLearningCurve = 1;
learnAlsWr = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda, displayLearningCurve);

name = 'ALSWR';
[e.tr.(name), e.te.(name)] = evaluate(name, learnAlsWr);

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

%% Top-K recommendation (on the full dataset using dim-reduction)
% TODO: select K with cross-validation
K = 150;
% Parameters for ALS-WR
% Our goal here is to obtain a version of Ytrain but with lower
% dimensionality. We're not trying to predict from the result, so we
% can overfit completely.
nFeatures = 200;
lambda = 0.000001;

reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 0)';
getSimilarity = @(Ytrain, Ytest, userDV) computeSimilarityMatrix(Ytrain, Ytest, userDV, reduceSpace);
learnTopKALS = @(Y, Ytest, userDV, artistDV) ...
    learnTopKPredictor(Y, Ytest, userDV, artistDV, K, getSimilarity(Y, Ytest, userDV));

name = ['Top', int2str(K), 'NeighborsALS'];
[e.tr.(name), e.te.(name)] = evaluate(name, learnTopKALS);

clearvars K nFeatures lambda;

%% Top-K recommendation with Fisher Transform
% Doesn't seem to change anything.
%{
learnTopKFisher = @(Y, Ytest, userDV, artistDV) learnTopKPredictor(Y, Ytest, userDV, artistDV, K, Sfisher);

name = ['Top', int2str(K), 'NeighborsALSFisher'];
[e.tr.(name), e.te.(name)] = evaluate(name, learnTopKFisher);
fprintf('RMSE with Top-K recommendation (K = %d) with Fisher transform: %f | %f\n', K, e.tr.topKFisher, e.te.topKFisher);
%}

%% Gaussian Mixture Model clustering (soft clustering)
% TODO

%% Other predictions
% TODO
