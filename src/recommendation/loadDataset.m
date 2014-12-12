%% Dataset pre-processing

% Cell array holding the errors made with various methods
e = {};
e.tr = {}; e.te = {};

% Load dataset
load('./data/recommendation/songTrain.mat');

% Listening counts matrix:
% Each (i, j) corresponds to the listening count of user i for artist j
Yoriginal = Ytrain;
Goriginal = Gtrain;
clear artistName;

% TODO: when predicting the actual target data, first check if the pair is
% available in the training data. It yields 0 error for free!

%% Outliers removal
% TODO: test removing more or less "outliers"
nDev = 3;
Y = removeOutliersSparse(Yoriginal, nDev);
clearvars nDev;

%% Train / test split
% TODO: average over many train / test splits (use bootstrapping?)
setSeed(1);
% TODO: vary test / train proportions
[~, Ytest, ~, Ytrain, Gtrain] = splitData(Y, Goriginal, 0, 0.1);
[idx, sz] = getRelevantIndices(Ytrain, Ytest);
[testIdx, testSz] = getRelevantIndices(Ytest);
