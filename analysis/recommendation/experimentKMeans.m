function [trError, teError, assignments] = experimentKMeans(Yoriginal, Goriginal, assignments)
% EXPERIMENTKMEANS
    %{
    addpath(genpath('./data'), genpath('../data'));
    addpath(genpath('./src'), genpath('../src'));
    clearvars;

    loadDataset;
    %}
    setSeed(1);
    [~, Ytest, ~, Ytrain, ~] = getTrainTestSplit(Yoriginal, Goriginal, 0.3, 0, 3);
    [userDV, artistDV] = generateDerivedVariables(Ytest);
    
    if(~exist('assignments', 'var'))
        K = 10;
        % Matlab's version is *way* too long (even in parallel)
        %options = statset('UseParallel', 1);
        %[assignments, centroids] = kmeans(Ytrain, K, 'Distance', 'sqeuclidean', 'Options', options);
        % Piotr's version runs super fast
        [assignments, ~, ~] = kmeans2(Ytrain, K, 'metric', 'cosine', 'display', 1);
    end;
    
    predictor = @(user, artist) ...
        predictFromVotes(user, artist, Ytrain, assignments, userDV, artistDV);
    
    [idx, sz] = getRelevantIndices(Ytrain);
    Yhat = predictCounts(predictor, idx, sz);
    trError = computeRmse(Ytrain, Yhat);
    
    [idx, sz] = getRelevantIndices(Ytest);
    YtestHat = predictCounts(predictor, idx, sz);
    teError = computeRmse(Ytest, YtestHat);
    
    diagnoseError(Ytrain, Yhat, Ytest, YtestHat);
end

function prediction = predictFromVotes(user, artist, Y, assignments, userDV, artistDV)
    usersInCluster = find(assignments == assignments(user));
    
    participants = usersInCluster(Y(usersInCluster, artist) ~= 0);
    if(~isempty(participants))
        % Take votes in cluster:
        %   deviation to participant's average + this user's average
        % TODO: use the distance to weight the prediction?
        votes = full(Y(participants, artist) - userDV(participants, 1));
        prediction = userDV(user, 1) + mean(votes);
    else
        % Not enough information available in this cluster
        prediction = userDV(user, 1);
    end;
end