%% Strong prediction
% Predict counts for unseen users, based on known artists
% and the friendship graph.
% This is known as "cold start".

loadDataset;

nSeeds = 5;

% TODO: use the automatic random train / test splits
setSeed(-1);
nDev = 1; % Be very conservative
strongRatio = 0.2;
[YtestStrong, Ytest, Gstrong, Ytrain, Gtrain] = ...
    getTrainTestSplit(Yoriginal, Goriginal, 0, strongRatio, nDev);

[idx, sz] = getRelevantIndices(YtestStrong);
[userDV, artistDV] = generateDerivedVariables(Ytrain);

clearvars nDev strongRatio;

%% Simply predict the mean of artists
% Although we have no idea what kind of volume this user would have
% It turns out this does worse than just predicting a constant overall!

name = 'ArtistMean';
% [e.st.(name)] = evaluate(name, @learnAveragePerArtistPredictor);
meanPredictor = learnAveragePerArtistPredictor(Ytrain, Ytest, userDV, artistDV);
YhatMean = predictCounts(meanPredictor, idx, sz);

diagnoseError(YtestStrong, YhatMean);
e.st.(name) = computeRmse(YtestStrong, YhatMean);
fprintf('----- %s [single run]: %f\n\n', name, e.st.(name));

%% Predict the overall mean
% Although we have no idea what kind of volume this user would have
% It turns out this does worse than just predicting a constant overall!

for i = 1:nSeeds
    [YtestStrong, Ytest, Gstrong, Ytrain, Gtrain] = ...
        getTrainTestSplit(Yoriginal, Goriginal, 0, 0.2, 2);
    [idx, sz] = getRelevantIndices(YtestStrong);
    
    name = 'Constant';
    overallMean = mean(nonzeros(Ytrain));
    constantPredictor = @(user, artist) overallMean;
    YhatConstant = predictCounts(constantPredictor, idx, sz);

    diagnoseError(YtestStrong, YhatConstant);
    e.st.(name) = computeRmse(YtestStrong, YhatConstant);
    fprintf('----- %s [single run]: %f\n\n', name, e.st.(name));
end;

%% Use artist derived variables
% E.g. overall mean + artist likeability

for i = 1:nSeeds
    [YtestStrong, Ytest, Gstrong, Ytrain, Gtrain] = ...
        getTrainTestSplit(Yoriginal, Goriginal, 0, 0.2, 2);
    [userDV, artistDV] = generateDerivedVariables(Ytrain);
    [idx, sz] = getRelevantIndices(YtestStrong);
    
    name = 'ArtistLike';
    artistLikePredictor = learnArtistBasedPredictor(Ytrain, Gstrong, userDV, artistDV);
    YhatArtistLike = predictCounts(artistLikePredictor, idx, sz);

    diagnoseError(YtestStrong, YhatArtistLike);
    e.st.(name) = computeRmse(YtestStrong, YhatArtistLike);
    fprintf('----- %s [split %d]: %f\n\n', name, i, e.st.(name));
end;

%% Leverage the social graph
% Trust at 100% votes from friends, when there are enough (fallback on
% means)
% TODO: try making broader clusters from the friendship graph?

for i = 1:nSeeds
    [YtestStrong, Ytest, Gstrong, Ytrain, Gtrain] = ...
        getTrainTestSplit(Yoriginal, Goriginal, 0, 0.2, 2);
    [userDV, artistDV] = generateDerivedVariables(Ytrain);
    [idx, sz] = getRelevantIndices(YtestStrong);
    
    name = 'BFFs';
    % [e.st.(name)] = evaluate(name, @learnAveragePerArtistPredictor);
    bffPredictor = learnFriendsPredictor(Ytrain, Gstrong, userDV, artistDV);
    YhatBFF = predictCounts(bffPredictor, idx, sz);

    diagnoseError(YtestStrong, YhatBFF);
    e.st.(name) = computeRmse(YtestStrong, YhatBFF);
    fprintf('----- %s [split %d]: %f\n\n', name, i, e.st.(name));
end;




%% Weighted combination
% TODO: choose automatically
alpha = 0.6;

