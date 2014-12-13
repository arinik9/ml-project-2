function betas = modelEachArtist(Y, G, headThreshold, userDV, artistDV)
% MODELEACHARTIST Train a linear model for each nonzero artist
%
% INPUT
%   Y: (n x d) Listening counts matrix
%   G: (n x n) Social network
% OUTPUT
%   betas: Learnt weights vector (one column per nonzero artist)
%         (generate predictions with tX * betas(j))

    [idx, sz] = getRelevantIndices(Y);

    % TODO: do not hardcode the number of extracted features
    betas = zeros(11 + 1, sz.a);

    for j = 1:length(idx.unique.a)
        artist = idx.unique.a(j);
        users = idx.u(idx.a == artist);

        y = nonzeros(Y(users, artist));

        % We're only responsible of learning models for the head
        if(length(users) >= headThreshold)
            % Train a linear model for artist j
            tX = generateFeatures(artist, users, G, userDV, artistDV);

            % Simple least squares
            %betas(:, artist) = (tX' * tX) \ (tX' * y);
            % Ridge regression
            % TODO: choose lambda by cross-validation
            % TODO: do better than a simple ridge regression
            lambda = 0.01;
            betas(:, artist) = ridgeRegression(y, tX, lambda);
        end;
    end;

end
