%% Use Top K recommendation to predict and analyze the error achieved
% TODO: refactor as a function
clearvars;
loadDataset;
[userDV, ~] = generateDerivedVariables(Ytrain);

% Select users who have listened to a lot of different artists (head)
topUsers = find(userDV(:, 2) > 38);
Ysmall = Ytrain(topUsers, :);
YtestSmall = Ytest(topUsers, :);

nYsmall = normalizedSparse(Ysmall);
[userDV, ~] = generateDerivedVariables(nYsmall);

%% Learn & predict
[idx, ~] = getRelevantIndices(Ysmall);
[idxTest, ~] = getRelevantIndices(YtestSmall);
K = 20;

% Avoid regenerating the similarity matrix
if(~exist('S', 'var'))
    [Yhat, S] = topKRecommendation(nYsmall, idx, K, userDV);
else
    Yhat = topKRecommendation(nYsmall, idx, K, userDV, S);
end;

YtestHat = topKRecommendation(nYsmall, idxTest, K, userDV, S);

%% Visualize error
diagnoseError(normalizedSparse(Ysmall), Yhat);
diagnoseError(normalizedSparse(YtestSmall), YtestHat);

%% Visualize predictions repartitions
figure;
subplot(1, 2, 1);
hist(nonzeros(normalizedSparse(YtestSmall)), 20);
title('Repartition of normalized test data');
subplot(1, 2, 2);
hist(nonzeros(YtestHat), 20);
title('Repartition of normalized predictions');