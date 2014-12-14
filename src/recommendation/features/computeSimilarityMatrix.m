function similarities = computeSimilarityMatrix(Y, Ytest, userDV, reduceDimensionality)
% COMPUTESIMILARITYMATRIX Generate the similarity matrix for all users
%
% INPUT
%   Y:            (users x artists)
%   Ytest:        (users x artists)
%   userDV:       precomputed Derived Variables
%   [reduceSpace] function(Y, Ytest) returning Y projected on a space of lower dimensionality
%                 The projected Y must still have one line per user, but fewer features.
%                 It is assumed to be dense.
% OUTPUT
%   similarities: (users x users) Symmetric matrix giving the similarity
%                 for each pair of users

    u = size(Y, 1);

    % TODO: is it necessary to make a difference?

    if(exist('reduceDimensionality', 'var'))
        % ----- Dense mode
        Yreduced = reduceDimensionality(Y, Ytest);
        Yreduced = normalizedDense(Yreduced);
        
        similarities = zeros(u, u);

        parfor_progress(u);
        parfor a = 1:u
            for b = 1:u
                % Only need to compute one half
                if (a > b)
                    similarities(a, b) = computeSimilarity(Yreduced, a, b, userDV);
                end;
            end;
            parfor_progress;
        end;
        parfor_progress(0);
        
    else
        % ----- Sparse mode
        listenedBy = getListenedBy(Y);

        % The similarity matrix is going to be very sparse
        values = [];
        idxa = [];
        idxb = [];
        parfor_progress(u);
        parfor a = 1:u
            for b = 1:u
                % Only need to compute one half
                if (a > b)
                    v = computeSimilaritySparse(Y, a, b, userDV, listenedBy);
                    if(abs(v) > eps)
                        values = [values; v];
                        idxa = [idxa; a];
                        idxb = [idxb; b];
                    end;
                end;
            end;
            
            parfor_progress;
        end;
        parfor_progress(0);
        
        similarities = sparse(idxa, idxb, values, u, u);
    end;

    % The similarity matrix is always symmetric
    similarities = similarities + similarities';
end

function similarity = computeSimilaritySparse(Y, a, b, userDV, listenedBy)
% COMPUTESIMILARITYSPARSE Pearson correlation
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
        means = [userDV(a, 1); userDV(b, 1)];
        similarity = computePearsonCoefficient(Ysub, means);
    else
        similarity = 0;
    end;
end

function similarity = computeSimilarity(Y, a, b, userDV)
% COMPUTESIMILARITY
% INPUT:
%   Y: Y is assumed to be dense
    Ysub = [Y(a, :); Y(b, :)];
    means = [userDV(a, 1); userDV(b, 1)];
    similarity = computePearsonCoefficient(Ysub, means);
end

function similarity = computePearsonCoefficient(Ysub, means)
    % Deviations of the common ratings to the average rating of the user
    Ydev = [Ysub(1, :) - means(1); Ysub(2, :) - means(2)];
    % Squared deviation to the average listening counts
    deviations = sum(Ydev .^ 2, 2);

    similarity = sum(Ydev(1, :) .* Ydev(2, :), 2) / sqrt(deviations(1) * deviations(2));
end

