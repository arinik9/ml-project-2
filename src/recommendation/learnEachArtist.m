function betas = learnEachArtist(Y, G, userDV, artistDV)
% LEARNEACHARTIST Train a linear model for each nonzero artist
%
% INPUT
%   Y: (n x d) Listening counts matrix
%   G: (n x n) Social network
% OUTPUT
%   betas: Learnt weights vector (one column per nonzero artist)
%         (generate predictions with tX * betas(j))

    [idx, sz] = getRelevantIndices(Y);

    % Head / tail split
    headThreshold = 100;
    % TODO: do not hardcode the number of extracted features
    betas = zeros(11 + 1, sz.a);

    for j = 1:length(idx.tr.unique.a)
        artist = idx.tr.unique.a(j);
        users = idx.tr.u(idx.tr.a == artist);

        y = nonzeros(Y(users, artist));

        if(length(users) > headThreshold)
            % Train a linear model for artist j
            tX = generateFeatures(artist, Y, G, userDV, artistDV);

            % Simple least squares
            betas(:, artist) = (tX' * tX) \ (tX' * y);
        end;
    end;

end
