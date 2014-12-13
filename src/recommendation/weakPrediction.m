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
nFeatures = 50; % Target reduced dimensionality
lambda = 0.05;
displayLearningCurve = 1;
learnAlsWr = @(Y, Ytrain, userDV, artistDV) learnAlsWrPredictor(Y, Ytrain, userDV, artistDV, nFeatures, lambda, displayLearningCurve);

[e.tr.als, e.te.als] = evaluate(learnAlsWr);
fprintf('RMSE ALS-WR (low rank): %f | %f\n', e.tr.als, e.te.als);

%% "Each Artist" predictions
% Train a separate model for each artist using derived variables
[e.tr.eachArtist, e.te.eachArtist] = evaluate(@learnEachArtistPredictor);
fprintf('RMSE with Each Artist (one model per artist) : %f | %f\n', e.tr.eachArtist, e.te.eachArtist);

%% Head / tail predictor
% Train a separate model for each artist of the head
% handle the tail differently
headThreshold = 100;
learnHeadTail = @(Y, Ytrain, userDV, artistDV) learnHeadTailPredictor(Y, Ytrain, userDV, artistDV, headThreshold);
[e.tr.eachArtist, e.te.eachArtist] = evaluate(learnHeadTail);
fprintf('RMSE with head / tail (threshold = %d) : %f | %f\n', headThreshold, e.tr.eachArtist, e.te.eachArtist);


%% Other predictions
% TODO
