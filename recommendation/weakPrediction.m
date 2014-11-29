%% Weak prediction on the song recommendation dataset
% Weak prediction: for an existing user, predict unobserved listening counts
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

%% Dataset pre-processing
% TODO: refactor

clearvars;

% Load dataset
load('./data/recommendation/songTrain.mat');

% Counts matrix:
% Each (i, j) corresponds to the listening count of user i for artist j
Yoriginal = Ytrain;
Goriginal = Gtrain;

%% Train / test split
% TODO: cross-validate all the things!
% TODO: no need to withold users for strong prediction if we're focusing on
% weak prediction here
[~, Ytest, ~, Ytrain, ~] = splitData(Yoriginal, Goriginal);
% Total size of train and test matrices
[trN, trD] = size(Ytrain);
[teN, teD] = size(Ytest);
% Indices of available counts
[trUserIndices, trArtistIndices] = find(Ytrain);
[teUserIndices, teArtistIndices] = find(Ytest);

