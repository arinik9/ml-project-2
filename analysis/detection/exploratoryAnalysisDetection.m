%% Detection Exploratory analysis
clearvars;

addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

% Load dataset
load('./data/detection/train_feats.mat');
% No need to load the images for now
%load('./data/detection/train_imgs.mat');

y = labels;

%% Generate feature vectors (so each one is a row of X)
fprintf('Generating feature vectors..\n');
D = numel(feats{1});  % feature dimensionality
X = zeros([size(feats, 1) D]);

% convert features to a vectors of D dimensions
for i=1:length(feats)
    X(i,:) = feats{i}(:); 
end;

% TODO : factorize dataset handling code

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
