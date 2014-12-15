function predictor = learnKMeansPredictor(Ytrain, Ytest, userDV, ~, K, S)
% LEARNKMEANSPREDICTOR Cluster
%
% INPUT
%   K: Maximum number of clusters (may not all be used)
%   S: Precomputed similarity matrix, used to weigh the neighbor's votes
%      It is assumed to be symmetric.

    if(~exist('K', 'var'))
        K = 15;
    end;

    % Matlab's version is *way* too long (even in parallel)
    % options = statset('UseParallel', 1);
    % [assignments, centroids] = kmeans(Ytrain, K, 'Distance', 'sqeuclidean', 'Options', options);

    % Piotr's version runs super fast
    [assignments, centroids, ~] = kmeans2(Ytrain, K, 'metric', 'cosine', 'display', 1);


    if(exist('S', 'var'))
        predictor = @(user, artist) ...
            predict(user, artist, Ytrain, assignments, centroids, userDV, S);
    else
        predictor = @(user, artist) ...
            predict(user, artist, Ytrain, assignments, centroids, userDV);
    end;
end

function prediction = predict(user, artist, Y, assignments, centroids, userDV, S)
    usersInCluster = find(assignments == assignments(user));
    participants = usersInCluster(Y(usersInCluster, artist) ~= 0 & usersInCluster ~= user);

    if(~isempty(participants))
        if(exist('S', 'var'))
            prediction = predictVotesWeightedBySimilarity(user, artist, Y, participants, userDV, S);
            %prediction = predictVotesWeightedByDistance(user, artist, Y, participants, userDV);
        else
            % Simply predict the cluster's centroid
            predictedVector = centroids(assignments(user), :);
            prediction = predictedVector(artist);
        end;
    else
        % Not enough information available in this cluster
        prediction = userDV(user, 1);
    end;
end
