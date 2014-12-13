function predictor = learnTopKPredictor(Y, Ytest, userDV, artistDV, K, S)
% LEARNTOPKPREDICTOR Use counts from the K most similar users to generate predictions
%
% INPUT
%   K: number of neighbors to use
%   S: precomputed similarity matrix

    K = min(K, size(Y, 1));
    % Use less than K neighbors if they're not similar enough
    similarityThreshold = 0.5;

    % Precompute useful values
    listenedBy = getListenedBy(Y);
    neighbors = precomputeTopKNeighbors(K, S, similarityThreshold);
    overallMean = mean(nonzeros(Y));

    predictor = @(user, artist) predict(Y, user, artist, neighbors, userDV, listenedBy, overallMean);
end

function prediction = predict(Y, user, artist, neighbors, userDV, listenedBy, overallMean)
% PREDICT Find the K most similar users similar to our target
% and predict entries based on their tastes
    if(~isempty(listenedBy{user}))
        prediction = aggregate(Y, user, artist, neighbors{user}, userDV);
    else
        % TODO: handle the "no neighbors" situation more gracefully
        prediction = 0;
    end;

    prediction = prediction + overallMean;
end

function prediction = aggregate(Y, user, artist, neighbors, userDV)
% AGGREGATE Use preferences of neighbors to predict the unseen count
%
% INPUT
%   Y:              (users x artists)
%   (user, artist): coordinates of the count to predict
%   neighbors:      list of neighbors of `user`
%   userDV:         precomputed Derived Variables
% OUTPUT
%   prediction:     a real (continuous) listening count prediction
%                   based on the observed counts from the user's neighbors

    % TODO: handle 0 neighbors case as gracefully as possible
    % TODO: if the count is already available, use it
    if(~isempty(neighbors))
        %fprintf('User %d has %d neighbors, worst similarity %f\n', user, size(neighbors, 1), neighbors(end));
        neighborsCounts = full(Y(neighbors(:, 1), artist));
        neighborsDev = neighborsCounts - userDV(neighbors(:, 1), 1);
        normalization = 1 / sum(abs(neighbors(:, 2)));

        prediction = userDV(user, 1) + normalization * sum(neighbors(:, 2) .* neighborsDev, 1);
    else
        prediction = userDV(user, 1);
    end;
end

function neighbors = precomputeTopKNeighbors(K, S, similarityThreshold)
%
% OUTPUT
%   neighbors: cell array (one entry per user)
%              with a list [index, similarity measure] of the top K
%              neighbors of each user

    n = size(S, 1);
    neighbors = cell(n, 1);
    for user = 1:n
        % Get K most similar users from the similarity matrix
        [userSimilarities, userNeighbors] = sort(S(user, :), 'descend');
        % [index of neighbor, similarity measure]
        neighbors{user} = [full(userNeighbors(1:K))', full(userSimilarities(1:K))'];
        % Use less than K neighbors if they're not similar enough
        neighbors{user} = neighbors{user}(neighbors{user}(:, 2) > similarityThreshold, :);
    end;
end
