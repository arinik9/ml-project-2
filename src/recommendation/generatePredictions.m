% Generate the final predictions using the best learnt models
clearvars;
loadDataset;
load('./data/recommendation/songTestPairs.mat');
empty = [];

%% ---------- Weak generalization
% Given existing (user, artist) pairs

% Normalize train dataset
Ynorm = normalizedSparse(Ytrain);

[userDV, ~] = generateDerivedVariables(Ynorm);
if(~exist('S', 'var'))
    S = computeSimilarityMatrix(Ynorm, empty, userDV);
end;
weakPredictor = learnSimilarityBasedPredictor(Ynorm, empty, userDV, empty, S);

% Sanity check: train error
[idxTrain, szTrain] = getRelevantIndices(Ynorm);
Yhat = predictCounts(weakPredictor, idxTrain, szTrain);
frprintf('Train RMSE: %f', computeRmse(Ynorm, Yhat));
%%
% Generate actual predictions
[idxTarget, szTarget] = getRelevantIndices(Ytest_weak_pairs);
nnzTarget = nnz(Ytest_weak_pairs);

logYweak = predictCounts(weakPredictor, idxTarget, szTarget);

% Cheap trick: if we alreay know an entry, just predict it directly
assert(nnz((Ytrain ~= 0) & (Ytest_weak_pairs ~= 0)) == 0);
% Then it is not applicable, too bad

% Denormalize the output
logValues = nonzeros(logYweak);
Yweak = sparse(idxTarget.u, idxTarget.a, exp(logValues), szTarget.u, szTarget.a);

% Verify
assert(nnz(logYweak) == nnzTarget);
figure;
subplot(2, 2, 1); hist(nonzeros(Ynorm));
subplot(2, 2, 2); hist(nonzeros(Ytrain));
subplot(2, 2, 3); hist(nonzeros(logYweak));
subplot(2, 2, 4); hist(nonzeros(Yweak));

%% ---------- Strong generalization
% Given unknown users
strongPredictor = 0;

% TODO: (!!!) denormalize the output

%% Output
saveRecommendationPredictions(logYweak, Ystrong);
