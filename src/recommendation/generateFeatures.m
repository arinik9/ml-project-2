function [X] = generateFeatures(j, Y, G, userFeatures, artistFeatures)
% GENERATEFEATURES Generate a feature matrix for a given artist
%
% See `generateDerivedVariables`.
%
% INPUT:
%   j: index of the artist to generate features for
%   Y: (n x d) listening counts for all artists
%   G: (n x n) social network (friendship relationships between users)
%   userFeatures: (n x ..) precomputed features for each user
%   artistFeatures: (d x ..) precomputed features for each artist
%
% OUTPUT:
%   X: (m x d) feature matrix,
%      - m = number of users who have listened to the given artist
%      - d = number of extracted features (8 + 3 + ?)

    [uIdx, ~] = find(Y(:, j));
    m = length(uIdx);

    dUser = size(userFeatures, 2);
    dArtist = size(artistFeatures, 2);
    dSocial = 0;
    X = zeros(m, dUser + dArtist + dSocial);

    % Features corresponding to each user having listened to this artist
    X(:, 1:dUser) = userFeatures(uIdx, :);
    % Features corresponding to the artist are repeated in data example
    X(:, (dUser+1):(dUser+dArtist)) = ones(m, 1) * artistFeatures(j, :);

    % TODO: generate more features from the social graph
end
