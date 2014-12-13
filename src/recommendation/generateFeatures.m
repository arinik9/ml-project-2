function [X] = generateFeatures(j, uIdx, Y, G, userFeatures, artistFeatures)
% GENERATEFEATURES Generate a feature matrix for a given artist
%
% See `generateDerivedVariables`.
%
% INPUT:
%   j: index of the artist to generate features for
%   uIdx: indices of users who have listened to this artist
%   Y: (n x d) listening counts for all artists
%   G: (n x n) social network (friendship relationships between users)
%   userFeatures: (n x ..) precomputed features for each user
%   artistFeatures: (d x ..) precomputed features for each artist
%
% OUTPUT:
%   X: (m x d) feature matrix, including an intercept term
%      - m = number of users who have listened to the given artist
%      - d = number of extracted features (1+ 8 + 3 + ?)

    m = length(uIdx);

    dUser = size(userFeatures, 2);
    dArtist = size(artistFeatures, 2);
    dSocial = 0;
    X = zeros(m, 1 + dUser + dArtist + dSocial);
    X(:, 1) = 1;
    
    % Features corresponding to each user having listened to this artist
    X(:, 2:(dUser+1)) = userFeatures(uIdx, :);
    % Features corresponding to the artist are repeated in data example
    X(:, (dUser+2):(dUser+dArtist+1)) = ones(m, 1) * artistFeatures(j, :);

    % TODO: generate more features from the social graph
end
