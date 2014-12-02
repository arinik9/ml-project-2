clearvars;

% add path to source files and toolboxs
addpath(genpath('./toolbox/'));
addpath(genpath('./src/'));
addpath(genpath('./detection/'));

% Load both features and training images
load('./data/detection/train_feats.mat');
load('./data/detection/train_imgs.mat');

%% Pre-process data
fprintf('Generating feature vectors..\n');
X = generateFeatureVectors(feats);

% TODO: normalize

% TODO: do this randomly! and k-fold!
fprintf('Splitting into train/test..\n');
Tr.idxs = 1:2:size(X,1);
Tr.X = X(Tr.idxs,:);
Tr.y = labels(Tr.idxs);

Te.idxs = 2:2:size(X,1);
Te.X = X(Te.idxs,:);
Te.y = labels(Te.idxs);

%% Prediction using different models
fprintf('Predictions using different models..\n');

fprintf('Default NN prediction\n');
% Default NN prediction (tanh, learningRate=1, ...)
nnPred = neuralNetworkPredict(Tr, Te);

fprintf('Tuned NN prediction\n');
% Tuned NN prediction (sigmoid activations and lower learningRate)
nnPred2 = neuralNetworkPredict(Tr, Te, 0, 1, 'sigm');

fprintf('Random prediction\n');
% Random prediction
randPred = rand(size(Te.y)); 


%% See prediction performance
fprintf('Plotting performance..\n');

% labels used to evaluate multiple methods
trueLabels = Te.y > 0;

% Methods names for legend
methodNames = {'NN Sigmoid Activation', 'NN TanH Activation', 'Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( trueLabels, [nnPred,nnPred2, randPred], true, methodNames );

avgTPRList