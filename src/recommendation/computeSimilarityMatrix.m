function similarities = computeSimilarityMatrix(Y, userDV)
% COMPUTESIMILARITYMATRIX Generate the similarity matrix for all users
%
% INPUT
%   Y:            (users x artists)
%   userDV:       precomputed Derived Variables
% OUTPUT
%   similarities: (users x users) Symmetric matrix giving the similarity
%                 for each pair of users

    u = size(Y, 1);

    listenedBy = getListenedBy(Y);

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
