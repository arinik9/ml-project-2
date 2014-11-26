%% Detection Exploratory analysis
clearvars;

addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

% Load both features and training images
load('./data/detection/train_feats.mat');
load('./data/detection/train_imgs.mat');

y = labels;

%% -- Generate feature vectors (so each one is a row of X)
fprintf('Generating feature vectors..\n');
D = numel(feats{1});  % feature dimensionality
X = zeros([length(imgs) D]);

for i=1:length(imgs)
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end;

% TO DO : factorize code

%% Output visualization

N = length(y);

% class selector
t = y > 0;
length(y(t))
length(y(~t))

% We have 8545 images: 
% - 1237 positive (with people) 
% - 7308 negative (without people)

hist(y);

%% Features visualization

% Normalize the data
[Xnorm, mu, sigma] = zscore(X); % X, get mu and std
X = Xnorm;

