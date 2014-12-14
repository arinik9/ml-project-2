function [trError, teError, assignments, S] = experimentKMeans(Ytrain, Ytest, userDV, assignments, S)
% EXPERIMENTKMEANS

    if(~exist('assignments', 'var'))
        K = 15;
        % Matlab's version is *way* too long (even in parallel)
        %options = statset('UseParallel', 1);
        %[assignments, centroids] = kmeans(Ytrain, K, 'Distance', 'sqeuclidean', 'Options', options);
        % Piotr's version runs super fast
        [assignments, ~, ~] = kmeans2(Ytrain, K, 'metric', 'cosine', 'display', 1);
    end;
    
    % Precompute similarity matrix
    if(~exist('S', 'var'))
        nFeatures = 200;
        lambda = 0.000001;

        reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 0, 2)';
        fprintf('Computing similarity matrix...\n');
        S = computeSimilarityMatrix(Ytrain, Ytest, userDV, reduceSpace);
        fprintf('Similarity matrix done!\n');
    end;
        
    predictor = @(user, artist) ...
        predictFromVotes(user, artist, Ytrain, assignments, userDV, S);
    
    [idx, sz] = getRelevantIndices(Ytrain);
    Yhat = predictCounts(predictor, idx, sz);
    trError = computeRmse(Ytrain, Yhat);
    
    [idx, sz] = getRelevantIndices(Ytest);
    YtestHat = predictCounts(predictor, idx, sz);
    teError = computeRmse(Ytest, YtestHat);
    
    diagnoseError(Ytrain, Yhat, Ytest, YtestHat);
end

function prediction = predictFromVotes(user, artist, Y, assignments, userDV, S)
    usersInCluster = find(assignments == assignments(user));
    participants = usersInCluster(Y(usersInCluster, artist) ~= 0 & usersInCluster ~= user);
    
    if(~isempty(participants))
        % Voting:
        %   deviation to participant's average + this user's average
        votes = full(Y(participants, artist) - userDV(participants, 1));
        
        % Weight vote of each user by its similarity
        % TODO: Fisher transform on the similarities?
        similarities = S(participants, user);
        similarities = similarities ./ sum(similarities);
        
        prediction = userDV(user, 1) + sum(similarities .* votes);
    else
        % Not enough information available in this cluster
        prediction = userDV(user, 1);
    end;
end