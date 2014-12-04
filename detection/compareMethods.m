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

%% Logistic regression

[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

Tr.tX = [ones(size(Tr.normX,1),1), Tr.normX];
Te.tX = [ones(size(Te.normX,1),1), Te.normX];

alpha = 0.5;
lambda = 10000;

beta = penLogisticRegression(Tr.y, Tr.tX, alpha, lambda);
%yHatLR = 

%% Prediction with different NN
fprintf('Predictions using different Neural Networks..\n');

fprintf('Default NN prediction\n');
% Default NN prediction (tanh, learningRate=1, ...)
%nnPred = neuralNetworkPredict(Tr, Te);

fprintf('Tuned NN prediction\n');
% Tuned NN prediction (sigmoid activations and lower learningRate)
%nnPred = neuralNetworkPredict(Tr, Te, 0, 1, 'sigm');

% Tuned NN prediction with Dropout Fraction set to 0.5 (close to optimality
% according to paper on dropout)
nnPred3 = neuralNetworkPredict(Tr, Te, 0, 1, 'sigm', 0, 1e-4);

% Tunned NN predictions with Weight Decay on L2 (Tikhonov regularization)
%nnPred3 = neuralNetworkPredict(Tr, Te, 0, 1, 'sigm', 0, 1e-4);

%% GP prediction
% TODO: Continue

meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEiso; ell = 1.0; sf = 1.0; hyp.cov = log([ell sf]);
likfunc = @likErf;

n = length(Te.X(1:5,:));

hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, Tr.X(1:5,:), Tr.y(1:5));
[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, Tr.X(1:5,:), Tr.y(1:5), Te.X(1:5,:), ones(n, 1));


%% Random Predictions

fprintf('Random prediction\n');
% Random prediction
randPred = rand(size(Te.y)); 


%% See prediction performance
fprintf('Plotting performance..\n');

% labels used to evaluate multiple methods
trueLabels = Te.y > 0;

% Methods names for legend
methodNames = {'Log Reg','Pen Log Reg','Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( trueLabels, [nnPred2, nnPred3, randPred], true, methodNames );

avgTPRList