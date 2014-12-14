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
    [assignments, ~, ~] = kmeans2(Ytrain, K, 'metric', 'cosine', 'display', 1);

    predictor = @(user, artist) ...
        predict(user, artist, Ytrain, assignments, userDV, S);
end

function prediction = predict(user, artist, Y, assignments, userDV, S)
    usersInCluster = find(assignments == assignments(user));
    participants = usersInCluster(Y(usersInCluster, artist) ~= 0 & usersInCluster ~= user);

    if(~isempty(participants))
        prediction = predictVotesWeightedBySimilarity(user, artist, Y, participants, userDV, S);
        %prediction = predictVotesWeightedByDistance(user, artist, Y, participants, userDV);
    else
        % Not enough information available in this cluster
        prediction = userDV(user, 1);
    end;
end

function prediction = predictVotesWeightedBySimilarity(user, artist, Y, participants, userDV, S)
% prediction = deviation to participant's average + this user's average
%              * normalized similarity measure as a way to indicate trust
    % TODO: this function is duplicated

    % TODO: try taking the vote directly
    votes = full(Y(participants, artist) - userDV(participants, 1));
    % Prediction is centered around this user's mean count
    prediction = userDV(user, 1);

    % Weight vote of each user by its similarity
    % TODO: Fisher transform on the similarities?
    similarities = S(participants, user);
    if(sum(abs(similarities)) > eps)
        % Normalize the similarity to use them as weights
        similarities = similarities ./ sum(abs(similarities));

        prediction = prediction + sum(similarities .* votes);
    end;
end
