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
diagnoseError(Ynorm, Yhat);
fprintf('[Weak] Train RMSE: %f\n', computeRmse(Ynorm, Yhat));
%%
% Generate actual predictions
[idxTarget, szTarget] = getRelevantIndices(Ytest_weak_pairs);
nnzTarget = nnz(Ytest_weak_pairs);

logYweak = predictCounts(weakPredictor, idxTarget, szTarget);

% Cheap trick: if we alreay know an entry, just predict it directly
assert(nnz((Ytrain ~= 0) & (Ytest_weak_pairs ~= 0)) == 0);
% Then it is not applicable, too bad

% Denormalize the output
valuesFloat = exp(nonzeros(logYweak));
values = ceil(valuesFloat); % Listening counts are integer valued
Yweak = sparse(idxTarget.u, idxTarget.a, values, szTarget.u, szTarget.a);

% Verify
assert(nnz(Yweak) == nnzTarget);
figure;
subplot(2, 2, 1); hist(nonzeros(Ynorm));
subplot(2, 2, 2); hist(nonzeros(Ytrain));
subplot(2, 2, 3); hist(nonzeros(logYweak));
subplot(2, 2, 4); hist(nonzeros(Yweak));

%% ---------- Strong generalization
% Given unknown users
Ynorm = normalizedSparse(Ytrain);
[userDV, artistDV] = generateDerivedVariables(Ynorm);

strongPredictor = learnArtistBasedPredictor(Ynorm, Goriginal, userDV, artistDV);

% Sanity check: train error
[idxTrainStrong, szTrainStrong] = getRelevantIndices(Ynorm);
YhatStrong = predictCounts(strongPredictor, idxTrainStrong, szTrainStrong);
diagnoseError(Ynorm, YhatStrong);
fprintf('[Strong] Train RMSE: %f\n', computeRmse(Ynorm, YhatStrong));
%%
% Generate actual predictions
[idxTargetStrong, szTargetStrong] = getRelevantIndices(Ytest_strong_pairs);
nnzTargetStrong = nnz(Ytest_strong_pairs);

logYstrong = predictCounts(strongPredictor, idxTargetStrong, szTargetStrong);

valuesStrongFloat = exp(nonzeros(logYstrong));
valuesStrong = ceil(valuesStrongFloat);
Ystrong = sparse(idxTargetStrong.u, idxTargetStrong.a, valuesStrong, szTargetStrong.u, szTargetStrong.a);

% Verify
assert(nnz(Ystrong) == nnzTargetStrong);
figure;
subplot(2, 2, 1); hist(nonzeros(Ynorm));
subplot(2, 2, 2); hist(nonzeros(Ytrain));
subplot(2, 2, 3); hist(nonzeros(logYstrong));
subplot(2, 2, 4); hist(nonzeros(Ystrong));


%% Output
saveRecommendationPredictions(Yweak, Ystrong);

%%
testRecommendationPredictions;
