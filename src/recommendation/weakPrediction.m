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

% Predict counts based on the artist's average listening count
% (which is one of the derived variables)
[e.tr.mean, e.te.mean] = evaluate(@learnAveragePerArtistPredictor);
fprintf('RMSE with a constant predictor per artist: %f | %f\n', e.tr.mean, e.te.mean);

%% ALS-WR
% TODO: experiment different lambdas and number of features
% TODO: choose the number of features by cross-validation
nFeatures = 50; % Target reduced dimensionality
lambda = 0.05;
displayLearningCurve = 1;

[U, M] = alswr(Ytrain, Ytest, nFeatures, lambda, displayLearningCurve);

e.tr.als = computeRmse(Ytrain, reconstructFromLowRank(U, M, idx, sz));
e.te.als = computeRmse(Ytest, reconstructFromLowRank(U, M, testIdx, testSz));

fprintf('RMSE ALS-WR (low rank): %f | %f\n', e.tr.als, e.te.als);
diagnoseError(Ytrain, reconstructFromLowRank(U, M, idx, sz));
diagnoseError(Ytest, reconstructFromLowRank(U, M, testIdx, testSz));

% Cleanup
clearvars nFeatures lambda displayLearningCurve; % U M

%% "Each Artist" predictions
headThreshold = 100;
% Train a model for each item using derived variables
% TODO: handle tail as well!
betas = learnEachArtist(Ytrain, Gtrain, headThreshold, userDV, artistDV);
%%
% Generage predictions
trPrediction = zeros(sz.tr.nnz, 1);
tePrediction = zeros(sz.te.nnz, 1);
for j = 1:sz.tr.unique.a
    artist = idx.tr.unique.a(j);
    users = idx.tr.u(idx.tr.a == artist);

    % TODO: are we generating the predictions correctly here?
    tXtrain = generateFeatures(artist, users, Ytrain, Gtrain, userDV, artistDV);
    % Since Xtrain has exactly as many lines as users listened to this
    % artist, this will automatically produce only the predictions we need
    trPrediction(idx.tr.a == artist) = tXtrain * betas(:, artist);


    usersTest = idx.te.u(idx.te.a == artist);
    if(~isempty(usersTest))
        tXtest = generateFeatures(artist, usersTest, Ytest, Gtrain, userDV, artistDV);

        tePrediction(idx.te.a == artist) = tXtest * betas(:, artist);
    end;
end;

trYhatLS = sparse(idx.tr.u, idx.tr.a, trPrediction, sz.tr.u, sz.tr.a);
teYhatLS = sparse(idx.te.u, idx.te.a, tePrediction, sz.te.u, sz.te.a);

e.tr.leastSquares = computeRmse(Ytrain, trYhatLS);
e.te.leastSquares = computeRmse(Ytest, teYhatLS);

fprintf('RMSE with Least-Squares on head only : %f | %f\n', e.tr.leastSquares, e.te.leastSquares);
diagnoseError(Ytrain, trYhatLS);

% Cleanup
clearvars j artist beta Xtrain tXtrain Xtest tXtest trPrediction tePrediction trVal teVal;
clearvars yTrain yTest trYhatLS teYhatLS;

%% Other predictions
% TODO
