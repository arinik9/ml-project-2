%% Detection Exploratory analysis
clearvars;

addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));
addpath(genpath('./detection'), genpath('../detection'));


% Load dataset
load('./data/detection/train_feats.mat');
% No need to load the images for now
%load('./data/detection/train_imgs.mat');

y = labels;

%% Generate feature vectors (so each one is a row of X)
fprintf('Generating feature vectors..\n');
X = generateFeatureVectors(feats);

%% Remove outliers

[Xnew, ynew] = removeOutliers(X, y, 3);

% It seems to us that there are no obvious outliers. All the data is at
% most at 3 times the deviation from the median (and only 4 datapoints are less than 2
% times the deviation)

%% Normalize the data
[Xnorm, mu, sigma] = zscore(X);
X = Xnorm;


%% Output (labels) visualization

N = length(y);

% class selector
t = y > 0;
length(y(t))
length(y(~t))

% We have 8545 images: 
% - 1237 positive (with people) 
% - 7308 negative (without people)
% We will need to be careful not to let this skewed distribution influence
% our results.
hist(y);

%% Features analysis

% Find features highly correlated to the output
threshold = 0.5;
selector = @(x) abs(x) > threshold;
[correlatedToOutput, correlations] = findCorrelations(selector, X, y);

% No feature explains much of the output by itself
size(correlatedToOutput, 1)

% Find features highly correlated among themselves
threshold = 0.90;
selector = @(x) abs(x) > threshold;
[correlatedVariables, correlations] = findCorrelations(selector, X);
% A *lot* of features are very highly correlated.
% We have highly redudant input. We need to apply dimensionality reduction.
size(correlatedVariables, 1)

%% 
