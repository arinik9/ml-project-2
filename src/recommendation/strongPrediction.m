%% Strong prediction
% Predict counts for unseen users, based on known artists
% and the friendship graph.
% This is known as "cold start".

loadDataset;

% TODO: use the automatic random train / test splits
setSeed(1);
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

name = 'Constant';
% [e.st.(name)] = evaluate(name, @learnAveragePerArtistPredictor);
overallMean = mean(nonzeros(Ytrain));
constantPredictor = @(user, artist) overallMean;
YhatConstant = predictCounts(constantPredictor, idx, sz);

diagnoseError(YtestStrong, YhatConstant);
e.st.(name) = computeRmse(YtestStrong, YhatConstant);
fprintf('----- %s [single run]: %f\n\n', name, e.st.(name));

%% Use artist derived variables
% E.g. overall mean + artist likeability
% TODO

%% Leverage the social graph
% Trust at 100% votes from friends, when there are enough (fallback on
% means)
% TODO: determine if the friendship graph actually helps
% TODO: try making broader clusters from the friendship graph?
name = 'BFFs';
% [e.st.(name)] = evaluate(name, @learnAveragePerArtistPredictor);
bffPredictor = learnFriendsPredictor(Ytrain, Gstrong, userDV, artistDV);
YhatBFF = predictCounts(bffPredictor, idx, sz);

diagnoseError(YtestStrong, YhatBFF);
e.st.(name) = computeRmse(YtestStrong, YhatBFF);
fprintf('----- %s [single run]: %f\n\n', name, e.st.(name));


%% Weighted combination
% TODO: choose automatically
alpha = 0.6;

