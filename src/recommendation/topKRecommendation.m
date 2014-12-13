function [Yhat, S] = topKRecommendation(Y, idx, K, userDV, S)
% TOPNRECOMMENDATION Use counts from the K most similar users to generate predictions
%
% INPUT
%   Y:      (users x artists)
%   idx:    Indices to predict on
%   K:      number of neighbors to use
%   userDV: precomputed Derived Variables
%   S:      [optional] precomputed similarity matrix
% OUTPUT
%   Yhat:   Predictions generated for all indices in `idx` (sparse matrix)
%   S:      The similarity matrix (can be kept for future use)

    K = min(K, size(Y, 1));
    % Use less than K neighbors if they're not similar enough
    similarityThreshold = 0.1;

    % Precompute useful values
    listenedBy = getListenedBy(Y);

    % Matrix of similarities between each pair of user
    % TODO: compute similarity in Principal Components space to speed up
    if(~exist('S', 'var'))
        fprintf('Computing similarity matrix between %d users...', size(Y, 1));
        S = computeSimilarityMatrix(Y, userDV, listenedBy);
        fprintf(' done.\n');
    end;

    % For each user, find the K most similar
    % and predict entries based on their tastes
    [~, sz] = getRelevantIndices(Y);

    values = zeros(length(idx.u), 1);
    for user = 1:sz.u
        if(~isempty(listenedBy{user}))
            % Get K most similar users from the similarity matrix
            [userSimilarities, userNeighbors] = sort(S(user, :), 'descend');
            % [index of neighbor, similarity measure]
            neighbors = [full(userNeighbors(1:K))', full(userSimilarities(1:K))'];
            % Use less than K neighbors if they're not similar enough
            neighbors = neighbors(neighbors(:, 2) > similarityThreshold, :);

            % For each artist to predict
            jj = idx.a(idx.u == user);
            for j = 1:length(jj)
                % Place the value carefully
                artist = jj(j);
                selector = (idx.u == user) & (idx.a == artist);

                values(selector) = aggregate(Y, user, artist, neighbors, userDV);
                % TODO: fix aggregate to prevent negative predictions
                % Or use centered data to leverage this?
            end;
        end;
    end;

    overallMean = mean(nonzeros(Y));
    Yhat = sparse(idx.u, idx.a, values + overallMean, sz.u, sz.a);
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

function similarities = computeSimilarityMatrix(Y, userDV, listenedBy)
% COMPUTESIMILARITYMATRIX Generate the similarity matrix for all users
%
% INPUT
%   Y:            (users x artists)
%   userDV:       precomputed Derived Variables
%   listenedBy:   precomputed list of artists listened by each user
% OUTPUT
%   similarities: (users x users) Symmetric matrix giving the similarity
%                 for each pair of users
    u = size(Y, 1);

    % The similarity matrix is going to be very sparse
    values = [];
    idxa = [];
    idxb = [];
    for a = 1:u
        for b = 1:u
            % Only need to compute one half
            if (a > b)
                v = computeSimilarity(Y, a, b, userDV, listenedBy);
                if(abs(v) > eps)
                    values = [values; v];
                    idxa = [idxa; a];
                    idxb = [idxb; b];
                end;
            end;
        end;
    end;

    similarities = sparse(idxa, idxb, values, u, u);
    similarities = similarities + similarities';
end

function similarity = computeSimilarity(Y, a, b, userDV, listenedBy)
% COMPUTESIMILARITY Pearson correlation
% Compute the similarity between users a and b
% based on artists that they have both listened to.
%
% Note this similarity measure is symmetric.
%
% INPUT
%   Y:          (users x artists)
%   a:          first user to compare (index)
%   b:          second user to compare (index)
%   userDV:     precomputed Derived Variables
%   listenedBy: precomputed list of artists listened by each user
% OUTPUT
%   similarity: the Pearson correlation similarity measure between
%               users `a` and `b` (value in [-1, 1])
% SEE ALSO
%   http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient

    % Get common counts
    common = intersect(listenedBy{a}, listenedBy{b});
    c = length(common);

    if(c > 0)
        Ysub = [full(Y(a, common)); full(Y(b, common))];

        % Deviations of the common ratings to the average rating of the user
        Ydev = [Ysub(1, :) - userDV(a, 1); Ysub(2, :) - userDV(b, 1)];
        % Squared deviation to the average listening counts
        deviations = sum(Ydev .^ 2, 2);

        similarity = sum(Ydev(1, :) .* Ydev(2, :), 2) / sqrt(deviations(1) * deviations(2));
    else
        similarity = 0;
    end;
end

function listenedBy = getListenedBy(Y)
% GETLISTENEDBY
%
% INPUT
%   Y:          (users x artists)
% OUTPUT
%   listenedBy: Cell array (one entry per user) giving a list of the
%               of the artists this user listened to at least once

    u = size(Y, 1);
    listenedBy = cell(u, 1);
    for i = 1:u
        listenedBy{i} = find(Y(i, :));
    end;
end
