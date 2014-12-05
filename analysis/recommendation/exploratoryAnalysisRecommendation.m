%% Detection Exploratory analysis
clearvars;

addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

% Load dataset
load('./data/recommendation/songTrain.mat');

% Counts matrix:
% Each (i, j) corresponds to the listening count of user i for artist j
Y = Ytrain;
allCounts = nonzeros(Y);
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
[minCount, idxMin] = min(allCounts);
[maxCount, idxMax] = max(allCounts);
maxCount

% There seems to be a major outlier:
% (user 385, artist 9162) = 352698
% This is a reminder that we should remove outliers.

% Total listening count for an artist spans from 1 to 2274039
countsPerArtist = sum(Y(:, artistsIdx), 1); % Only for nonzero artists
[minArtistCount, idxMin] = min(countsPerArtist);
[maxArtistCount, idxMax] = max(countsPerArtist);
maxArtistCount
% The single most "popular" artist is Britney Spears

% Simple model: average count for an artist / average count for all artists
% This is a simple estimate of popularity

clear minCount minArtistCount idxMin idxMax;

%% Simple outliers removal
% Allow up to `nDev` deviations from the median
nDev = 3;
nnzBefore = nnz(Y);
Y = removeOutliersSparse(Y, nDev);
disp(['We removed ', int2str(nnzBefore - nnz(Y)), ' outliers from Y']);

% Update the indices
allCounts = nonzeros(Y);
[uIdx, aIdx] = find(Y);
usersIdx = unique(uIdx);
artistsIdx = unique(aIdx);

% TODO: also need to handle artists who have 0 listenings

%% Normalization
% "Make the data more Gaussian"
% Counts are positive integers only, so it would violate our Gaussian
% distribution assumption.

[Ynormalized, meanPerUser, devPerUser] = normalizedByUsers(Y);

%% Verify the result of normalization

% Mean should be 0, deviation should be 1
for i = 1:length(usersIdx)
    if(nnz(Ynormalized(i, :)) > 1)
        counts = nonzeros(Ynormalized(i, :));
        assert(abs(mean(counts) - 0) < 1e-3);
        assert(abs(std(counts) - 1) < 1e-1);
    end;
end;

% --- Before
figure;
hist(nonzeros(Y), 20);
title({'Initial repartition of listening counts', ''});
%savePlot('./report/figures/recommendation/unnormalized-counts.pdf', 'Count', 'Occurrences');

% --- After
% We observe a nicer Gaussian distribution of the counts
% But we still have a "long tail" to the right.
figure;
hist(nonzeros(Ynormalized), 20);
title({'Repartition of listening counts after outliers', 'removal and normalization'});
%savePlot('./report/figures/recommendation/normalized-counts.pdf', 'Normalized count', 'Occurrences');

clear i counts;

%% Features analysis
% TODO
