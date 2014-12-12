function [YtestHat, S] = topKRecommendation(Y, Ytest, K, userDV, S)
% TOPNRECOMMENDATION Use counts from the K most similar users to generate predictions
%
% INPUT
%
% OUTPUT
%
    K = min(K, size(Y, 1));
    % Use less than K neighbors if they're not similar enough
    similarityThreshold = 0.1;
    
    % Precompute useful values
    listenedBy = getListenedBy(Y);

    % Matrix of similarities between each pair of user
    if(~exist('S', 'var'))
        fprintf('Computing similarity matrix between %d users...', size(Y, 1));
        S = computeSimilarityMatrix(Y, userDV, listenedBy);
        fprintf(' done.\n');
    end;
    
    % For each user, find the K most similar
    % and predict entries based on their tastes
    [idx, sz] = getRelevantIndices(Ytest);
    
    values = zeros(sz.nnz, 1);
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
                values(selector) = max(0, values(selector));
            end;
        end;
    end;
    
    YtestHat = sparse(idx.u, idx.a, values, sz.u, sz.a);
end

function prediction = aggregate(Y, user, artist, neighbors, userDV)
% AGGREGATE Use preferences of neighbors to predict the unseen count
    % TODO: handle 0 neighbors case as gracefully as possible
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
    u = size(Y, 1);

    % The similarity matrix is going to be very sparse
    values = [];
    idxa = [];
    idxb = [];
    for a = 1:u
        for b = 1:u
            if (a ~= b)
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
end

function similarity = computeSimilarity(Y, a, b, userDV, listenedBy)
% COMPUTESIMILARITY Pearson correlation
% Compute the similarity between users a and b
% based on artists that they have both listened to
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
% OUTPUT
%   listenedBy: Cell array (one entry per user) giving a list of indices
%               of the artists this user listened to at least once

    u = size(Y, 1);
    listenedBy = cell(u, 1);
    for i = 1:u
        listenedBy{i} = find(Y(i, :));
    end;
end
