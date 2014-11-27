%% Detection Exploratory analysis
clearvars;

addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

% Load dataset
load('./data/recommendation/songTrain.mat');

% Counts matrix:
% Each (i, j) corresponds to the listening count of user i for artist j
R = Ytrain;
[uIdx, aIdx] = find(R);
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

%% Normalization
% "Make the data more Gaussian"
% Counts are positive integers only, so it would violate our Gaussian
% distribution assumption.

% TODO

%% Artists visualization

% Listening counts for one (user, artist) span from 1 to 352698
[minCount, idxMin] = min(nonzeros(R))
[maxCount, idxMax] = max(nonzeros(R))

% There seems to be a major outlier:
% (user 385, artist 9162) = 352698
% This is a reminder that we should remove outliers.

% Total listening count for an artist spans from 1 to 2274039
countsPerArtist = sum(R(:, artistsIdx), 1); % Only for nonzero artists
[minArtistCount, idxMin] = min(countsPerArtist)
[maxArtistCount, idxMax] = max(countsPerArtist)
% The single most "popular" artist is Britney Spears

% Simple model: average count for an artist / average count for all artists
% This is a simple estimate of popularity

%% Features analysis

% TODO
