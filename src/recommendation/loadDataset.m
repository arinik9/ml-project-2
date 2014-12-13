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

%% Train / test split
% TODO: average over many train / test splits (use bootstrapping?)
setSeed(1);
% TODO: vary test / train proportions
[~, Ytest, ~, Ytrain, Gtrain] = splitData(Yoriginal, Goriginal, 0, 0.1);
[idx, sz] = getRelevantIndices(Ytrain, Ytest);
[testIdx, testSz] = getRelevantIndices(Ytest);

%% Normalization & outliers removal
Ytrain = normalizedSparse(Ytrain);
Ytest = normalizedSparse(Ytest);

% TODO: test removing more or less "outliers"
% TODO: remove outliers on the train set only
nDev = 3;
Ytrain = removeOutliersSparse(Ytrain, nDev);
clearvars nDev;
