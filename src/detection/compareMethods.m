clearvars;

% add path to source files and toolboxs
addpath(genpath('./toolbox/'));
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
Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

% if not using Piotr's toolbox (to replace normalize function)
% onesX = ones(size(Te.X,1), 1);
% Te.normX = (Te.X - onesX * mu) ./ (onesX * sigma); 


fprintf('Done! You can start playing with the features!\n');

%% Principal Component Analysis
fprintf('Performing Principal Component Analysis..\n');

% Matlab PCA but it is too slow with so many features
%[coeff, mu, eigenvalues] = pca(Tr.normX);
%[train, test] = prinCompProjection(coeff, Tr.normX(1:500,:), Te.normX(1:200,:), 50);

% Need to pass a DxN matrix with N # data example
[PCA.coeff, PCA.mu, PCA.latent] = pca(Tr.normX');


%% Plot PCA Representation
fprintf('Plotting percentage of the total variance explained by each principal component..\n');
% Might be useful to choose how many PC we are keeping.
% On lower dimension dataset we usually keep PC so that we have a
% cumulative percentage of 95% of the total variance but here it would be
% too many PC.

% Percentage of the total variance explained by each principal component
PCA.explained = PCA.latent ./sum(PCA.latent);

% Cumulative percentage of the total variance explained by each principal component
PCA.explainedCum = (cumsum(PCA.latent)./sum(PCA.latent));

% Are we plotting on every PC?
nPlot = size(PCA.explained,1) / 4;

e1 = PCA.explained(1:nPlot);
e2 = 100 * PCA.explainedCum(1:nPlot);

figure();
[ax,~,hLine] = plotyy(1:nPlot, e1, 1:nPlot, e2, 'bar', 'plot');
title('PCA on Features')
xlabel('Principal Component')
ylabel(ax(1),'Variance Explained per PC')
ylabel(ax(2),'Total Variance Explained (%)')
hLine.LineWidth = 3;
hLine.Color = [0,0.7,0.7];
ylim(ax(2),[1 100]);

clear e1 e2 nPlot ax hLine;

%% Apply PCA to train and test data
% Project train and test data in the space formed by the PC we have decided
% to keep. The features obtained are not normalized so we do normalize them 
% at the end

fprintf('Projecting train and test data into the lower space..\n');

% Number of PC kept (to choose)
PCA.kPC = 50;
fprintf('We are projecting on the first %d principal components..\n', PCA.kPC);

[pcaXTrain, pcaXhatTrain, pcaAvsqTrain] = pcaApply(Tr.normX', PCA.coeff, PCA.mu, PCA.kPC);
Tr.pcaX = pcaXTrain'; Tr.pcaXhatTrain = pcaXhatTrain'; Tr.pcaAvsq = pcaAvsqTrain;
[pcaXTest, pcaXhatTest, pcaAvsqTest] = pcaApply(Te.normX', PCA.coeff, PCA.mu, PCA.kPC);
Te.pcaX = pcaXTest'; Te.pcaXhatTrain = pcaXhatTest'; Te.pcaAvsq = pcaAvsqTest;

% Normalize reduced input features
[Tr.pcaX mu sigma] = zscore(Tr.pcaX);
Te.pcaX = normalize(Te.pcaX, mu, sigma);

clear pcaXTrain pcaXhatTrain pcaAvsqTrain pcaXTest pcaXhatTest pcaAvsqTest;

fprintf('Done! We have now reduced train and test set !\n');

%% Logistic Regression
% Logistic Regression implementation using NN: A simple layer NN is a
% logistic regression

logRegPred = neuralNetworkPredict(Tr.y, Tr.normX, Te.normX, 0, 1, 'sigm', 0, 0, [size(Tr.normX,2) 2]);

% We use values from the PCA
logRegPred = neuralNetworkPredict(Tr.y, Tr.pcaX, Te.pcaX, 0, 1, 'sigm', 0, 0, [size(Tr.pcaX,2) 2]);
logRegPredTrain = neuralNetworkPredict(Tr.y, Tr.pcaX, Tr.pcaX, 0, 1, 'sigm', 0, 0, [size(Tr.pcaX,2) 2]);


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
% nnPred3 = neuralNetworkPredict(Tr.y, Tr.normX, Te.normX, 0, 1, 'sigm', 0, 1e-4);
nn = learnNeuralNetwork(Tr.y, Tr.normX, 0, 1, 'sigm', 0, 1e-4);
nnPred = predictNeuralNetwork(nn, Te.normX);


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
nPlot = size(t,1);

tic;
% gpPred = GPClassificationPrediction(y,x,t);
gpModel = learnGPClassification(y, x);
gpPred = predictGPClassification(gpModel, t);
TimeSpent = toc;

yHatGP = outputLabelsFromPrediction(gpPred, 0.5);

% We do recover our large rate of negative VS positive images
hist(yHatGP);

%% Test GP
% Methods names for legend
methodNames = {'Logistic Regression','Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( Te.y > 0, [pred, randPred], true, methodNames );
%avgTPRListTr = evaluateMultipleMethods( Tr.y > 0, [nnPred3, randPredTrain], true, methodNames );

% TODO : Play with parameters

%% Random Predictions

fprintf('Random prediction\n');
% Random prediction
randPred = rand(size(Te.y)); 
randPredTrain = rand(size(Tr.y)); 

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