function Yhat = topKRecommendation(Y, K, userDV)
% TOPNRECOMMENDATION Use counts from the K most similar users to generate predictions

    % Matrix of similarities between each pair of user
    S = computeSimilarityMatrix(Y, userDV);

    % TODO
    Yhat = S; % sparse(size(Y, 1), size(Y, 2));
end

function similarities = computeSimilarityMatrix(Y, userDV)
    u = size(Y, 1);

    % Precompute useful values
    listenedBy = getListenedBy(Y);

    % The similarity matrix is going to be very sparse
    values = [];
    idxa = [];
    idxb = [];
    for a = 1:u
        for b = 1:u
            if (a ~= b)
                v = computeSimilarity(Y, a, b, userDV, listenedBy);
                if(v > eps)
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
