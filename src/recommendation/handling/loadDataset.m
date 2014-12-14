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
