%% Weak prediction on the song recommendation dataset
% Weak prediction: for an existing user, predict unobserved listening counts
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));
clearvars;

loadDataset;
[userDV, artistDV] = generateDerivedVariables(Ytrain);

% Shortcut
evaluate = @(learn) evaluateMethod(learn, Ytrain, Ytest, userDV, artistDV);

%% Baseline: constant predictor (overall mean of all observed counts)
[e.tr.constant, e.te.constant] = evaluate(@learnConstantPredictor);
fprintf('RMSE with a constant predictor: %f | %f\n', e.tr.constant, e.te.constant);

%% Simple model: predict the average listening count of the artist
% (which is one of the derived variables)
[e.tr.mean, e.te.mean] = evaluate(@learnAveragePerArtistPredictor);
fprintf('RMSE with a constant predictor per artist: %f | %f\n', e.tr.mean, e.te.mean);

%% ALS-WR
% TODO: experiment different lambdas and number of features
% TODO: select hyper-parameters by cross-validation
nFeatures = 100; % Target reduced dimensionality
lambda = 1;
displayLearningCurve = 1;
learnAlsWr = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda, displayLearningCurve);

[e.tr.als, e.te.als] = evaluate(learnAlsWr);
fprintf('RMSE ALS-WR (low rank): %f | %f\n', e.tr.als, e.te.als);

%% "Each Artist" predictions
% Train a separate model for each artist using derived variables
[e.tr.eachArtist, e.te.eachArtist] = evaluate(@learnEachArtistPredictor);
fprintf('RMSE with Each Artist (one model per artist): %f | %f\n', e.tr.eachArtist, e.te.eachArtist);

%% Head / tail predictor
% Train a separate model for each artist of the head
% Train a common model for each cluster of tail artists
headThreshold = 50;
learnHeadTail = @(Y, Ytest, userDV, artistDV) learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold);

[e.tr.eachArtist, e.te.eachArtist] = evaluate(learnHeadTail);
fprintf('RMSE with head / tail (threshold = %d): %f | %f\n', headThreshold, e.tr.eachArtist, e.te.eachArtist);

%% Top-K recommendation (on the full dataset using dim-reduction)
% TODO: select K with cross-validation
K = 50;

% Precompute the similarity matrix (only once)
if(~exist('S', 'var'))
    nFeatures = 20;
    lambda = 0.05;
    reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 1)';

    fprintf('Computing similarity matrix of %d users projected with ALS-WR...\n', size(Ytrain, 1));
    S = computeSimilarityMatrix(Ytrain, Ytest, userDV, reduceSpace);
    fprintf('Similarity matrix computation is done.\n');

    clearvars nFeatures lambda;
end;

learnTopKALS = @(Y, Ytest, userDV, artistDV) learnTopKPredictor(Y, Ytest, userDV, artistDV, K, S);
[e.tr.topKALS, e.te.topKALS] = evaluate(learnTopKALS);
fprintf('RMSE with Top-K recommendation (K = %d): %f | %f\n', K, e.tr.topKALS, e.te.topKALS);


%% Gaussian Mixture Model clustering
% (soft clustering)

%% Other predictions
% TODO
