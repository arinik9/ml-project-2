%% Detection Exploratory analysis
clearvars;

addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

% Load dataset
load('./data/recommendation/songTrain.mat');

% Counts matrix:
% Each (i, j) corresponds to the listening count of user i for artist j
Y = Ytrain;
[uIdx, aIdx] = find(Y);
% Indices of users having at least one data example
usersIdx = unique(uIdx);
% Indices of artists having at least one listen
artistsIdx = unique(aIdx);

% Note that R is "fat", we have:
% - 1774 users
% - 15082 artists
% The vast majority of (user, artist) counts are unobserved:
% we have 69617 observed ratings overs the 26755468 potential pairs.

% Friendship graph:
% Each (i, j) corresponds to a friendship relationship between users i and j
friendships = Gtrain;

%% Artists visualization

% Listening counts for one (user, artist) span from 1 to 352698
[minCount, idxMin] = min(nonzeros(Y))
[maxCount, idxMax] = max(nonzeros(Y))

% There seems to be a major outlier:
% (user 385, artist 9162) = 352698
% This is a reminder that we should remove outliers.

% Total listening count for an artist spans from 1 to 2274039
countsPerArtist = sum(Y(:, artistsIdx), 1); % Only for nonzero artists
[minArtistCount, idxMin] = min(countsPerArtist)
[maxArtistCount, idxMax] = max(countsPerArtist)
% The single most "popular" artist is Britney Spears

% Simple model: average count for an artist / average count for all artists
% This is a simple estimate of popularity

clear minCount minArtistCount idxMin idxMax;

%% Outliers removal
% TODO
% TODO: also need to handle artists who have 0 listenings

%% Normalization
% "Make the data more Gaussian"
% Counts are positive integers only, so it would violate our Gaussian
% distribution assumption.

% We normalize *among user*, that is we make users lie on the same scale (a
% heavy user will have counts comparable to a light user). This way, we
% retain the artists popularity, which is important information.
% TODO: is it the correct way? Wouldn't we want to know if the user is very
% active, in order to predict accurately? (If we keep the factors used at
% normalization, we can rescale afterwards)
meanPerUser = zeros(size(usersIdx, 1), 1);
deviationPerUser = zeros(size(usersIdx, 1), 1);
values = zeros(nnz(Y), 1);
zeroIdx = [];
for i = 1:length(usersIdx)
    counts = nonzeros(Y(i, :));
    meanPerUser(i) = mean(counts);
    deviationPerUser(i) = std(counts);
    if(length(counts) <= 1)
        zeroIdx = [zeroIdx; i];
    end;
end;
% If we have only 1 data point for a user, std() is 0
% (and we don't want to divide by 0).
% We replace by the deviation to the mean over all counts, as an estimate of
% the actual deviation we might have observed).
o = ones(nnz(zeroIdx), 1);
deviationPerUser(zeroIdx) = std([meanPerUser(zeroIdx)'; mean(meanPerUser) * o']);
meanPerUser(zeroIdx) = mean(meanPerUser) * o;

% Generate a new normalized sparse matrix
for i = 1:length(usersIdx)
    values(uIdx == usersIdx(i)) = (nonzeros(Y(i, :)) - meanPerUser(i)) ./ deviationPerUser(i);
end;
Ynormalized = sparse(uIdx, aIdx, values, size(Y, 1), size(Y, 2));

% TODO: apply the same factors to the test set
% TODO: denormalize after prediction to obtain the correct scale

clear i o counts thisUserIdx values;

%% Verify the result of normalization
normalizedCounts = nonzeros(Ynormalized);
% We observe a nicer Gaussian distribution
figure;

hist(normalizedCounts, 20);

% Mean should be 0, deviation should be 1
for i = 1:length(usersIdx)
    if(nnz(Ynormalized(i, :)) > 1)
        counts = nonzeros(Ynormalized(i, :));
        assert(abs(mean(counts) - 0) < 1e-3);
        assert(abs(std(counts) - 1) < 1e-1);
    end;
end;

clear i counts;

%% Features analysis
% TODO
