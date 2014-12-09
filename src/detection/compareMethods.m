clearvars;

% add path to source files and toolboxs
addpath(genpath('./toolbox/PiotrToolbox'));
addpath(genpath('./toolbox/DeepLearnToolbox'));
addpath(genpath('./toolbox/gpml-matlab-v3.4'));
addpath(genpath('./src/'));

% Load both features and training images
load('./data/detection/train_feats.mat');
% load('./data/detection/train_imgs.mat');

%% Pre-process data

fprintf('Generating feature vectors..\n');
X = generateFeatureVectors(feats);
y = labels;

% Split data into train and test set given a proportion
prop = 2/3;
fprintf('Splitting into train/test with proportion %.2f..\n', prop);
[Tr.X, Tr.y, Te.X, Te.y] = splitDataDetection(y, X, prop);

% Normalize
fprintf('Normalizing features..\n');
[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
% Te.normX = normalize(Te.X, mu, sigma);  % normalize test data
onesX = ones(size(Te.X,1), 1);
Te.normX = (Te.X - onesX * mu) ./ (onesX * sigma);


fprintf('Done! You can start playing with the features!\n');

%% Principal Component Analysis
fprintf('Performing Principal Component Analysis..\n');

% Matlab PCA but it is too slow with so many features
%[coeff, mu, eigenvalues] = pca(Tr.normX);
%[train, test] = prinCompProjection(coeff, Tr.normX(1:500,:), Te.normX(1:200,:), 50);

% Need to pass a DxN matrix with N # data example
[U, mu, vars] = pca(Tr.normX');

fprintf('Projecting train and test data into the lower space..\n');

% Number of PC kept
nPrinComp = 50;
fprintf('We are projecting on the first %d principal components..\n', nPrinComp);

[pcaXTrain, pcaXhatTrain, pcaAvsqTrain] = pcaApply(Tr.normX', U, mu, nPrinComp);
Tr.pcaX = pcaXTrain'; Tr.pcaXhatTrain = pcaXhatTrain'; Tr.pcaAvsq = pcaAvsqTrain;
[pcaXTest, pcaXhatTest, pcaAvsqTest] = pcaApply(Te.normX', U, mu, nPrinComp);
Te.pcaX = pcaXTest'; Te.pcaXhatTrain = pcaXhatTest'; Te.pcaAvsq = pcaAvsqTest;

% Normalize reduced input features
[Tr.pcaX muPCA sigmaPCA] = zscore(Tr.pcaX);
Te.pcaX = normalize(Te.pcaX, muPCA, sigmaPCA);

clear pcaXTrain pcaXhatTrain pcaAvsqTrain pcaXTest pcaXhatTest pcaAvsqTest;

% Plot on 1st and 2nd principal component
% figure()
% plot(score(:,1),score(:,2),'+')
% xlabel('1st Principal Component')
% ylabel('2nd Principal Component')

% figure()
% pareto(explained)
% xlabel('Principal Component')
% ylabel('Variance Explained (%)')

fprintf('Done! You''re now in a lower dimension space !\n');

%% Plot PCA Representation
% TODO: make it work using Piotr's PCA

% Are we plotting on every PC?
n = size(explained,1)/2;

e1 = explained(1:n);
e2 = expl(1:n);

figure();
[ax,hBar,hLine] = plotyy(1:n, e1, 1:n, e2, 'bar', 'plot');
title('PCA on Features')
xlabel('Principal Component')
ylabel(ax(1),'Variance Explained per PC')
ylabel(ax(2),'Total Variance Explained (%)')
hLine.LineWidth = 3;
hLine.Color = [0,0.7,0.7];
ylim(ax(2),[1 100]);

%% Select the principal component scores: the representation of X in the principal component space
% Old code: not needed anymore using Piotr's PCA

fprintf('Selecting the PCA scores..\n');

nPrinComp = 500;
Tr.Xpca = score(:,1:nPrinComp);
% Finding scores back (same result as selecting score column)
% Tr.Xpca = Tr.normX * coeff(:,1:nPrinComp);

% Compute scores of test data
pc = coeff(:,1:nPrinComp);
Te.Xpca = Te.normX * pc;

fprintf('Done! We have now reduced train and test set !\n');

%% Logistic regression: Not working
% We will use 1-layer NN to run logreg

%[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

Tr.tXpca = [ones(size(Tr.Xpca,1),1), Tr.Xpca];
%Te.tX = [ones(size(Te.normX,1),1), Te.normX];

alpha = 0.5;
lambda = 100;

beta = penLogisticRegressionAuto(Tr.y, Tr.pcaX);
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

%% GP large scale Classification

% Note: if the dimensionality is too big (very fat x matrix) we only get
% negative labels... -> PCA
% [Tr.pcaNormX muPCA sigmaPCA] = zscore(Tr.pcaX);
% Te.pcaNormX = normalize(Te.pcaX, muPCA, sigmaPCA);
x = Tr.pcaNormX;
y = Tr.y;
t = Te.pcaNormX;
n = size(t,1);

tic;
gpPred = GPClassificationPrediction(y,x,t);
TimeSpent = toc;

yHatGP = outputLabelsFromPrediction(gpPred, 0.5);

% We do recover our large rate of negative VS positive images
hist(yHatGP);

%% Test GP
% Methods names for legend
methodNames = {'GP Classification','Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( Te.y > 0, [gpPred, randPred], true, methodNames );

% TODO : Play with parameters

%% Random Predictions

fprintf('Random prediction\n');
% Random prediction
randPred = rand(size(Te.y)); 

%% Test on threshold choice
% TODO: how to plot a single point on ROC Curve for corresponding
% threshold?
% non log scale ROC curve is better to visualize threshold

yHatRandom = outputLabelsFromPrediction(randPred, 0.5);
yHatRandom2 = outputLabelsFromPrediction(randPred, 0.95);
[avgTPRR, auc] = fastROC(Te.y > 0, yHatRandom)
[avgTPRR2, auc2] = fastROC(Te.y > 0, yHatRandom2)

yHatnnPred2 = outputLabelsFromPrediction(nnPred2, 0.3);
[avgTPRnn, aucNN] = fastROC(Te.y > 0, yHatnnPred2)
yHatnnPred2bis = outputLabelsFromPrediction(nnPred2, 0.99);
[avgTPRnn, aucNN] = fastROC(Te.y > 0, yHatnnPred2bis)

% Methods names for legend
methodNames = {'0.5 threshold','0.95 threshold', 'NN 0.5 t', 'NN 0.56 t', 'NN proba'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( trueLabels, [yHatRandom, yHatRandom2, yHatnnPred2, yHatnnPred2bis, nnPred2], true, methodNames );


%% See prediction performance
fprintf('Plotting performance..\n');

% labels used to evaluate multiple methods
trueLabels = Te.y > 0;

% Methods names for legend
methodNames = {'Log Reg','Pen Log Reg','Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( trueLabels, [nnPred2, nnPred3, randPred], true, methodNames );

avgTPRList