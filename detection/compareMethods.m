% TODO: Clean file + factorize methods as nnPredict

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
% TODO: Continue : Take more data examples + normalize before + make
% fastROC work for result

% Give a try to GP regression as classification seems really slow. But is
% it a good way to proceed ?

x = Tr.X(1:100,:);
y = Tr.y(1:100);
z = Te.X(1:100,:);
n = size(x,1);

% Mean function
meanfunc = @meanConst; 
hyp.mean = 0;

% use inducing points u and to base the computations on cross-covariances 
% between training, test and inducing points only
nu = fix(n); iu = randperm(n); iu = iu(1:nu); u = x(iu,:);

% Covariance function
covfunc = @covSEiso; 
ell = 1.0; % characteristic length-scale
sf = 1.0; % standard deviation of the signal sf
hyp.cov = log([ell sf]);
covfuncF = {@covFITC, {covfunc}, u};

% To try also : 
% covSEard hyp.cov = log(ones(1, size(u,1) + 1) 1xD+1 size % we should try
% that one also

% Likelihood function
likfunc = @likGauss; 
sn = 0.1; % standard deviation of the noise
hyp.lik = log(sn); 

% compute the (joint) negative log probability (density) nlml (also called marginal likelihood or evidence)
% nlml = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y);

hyp = minimize(hyp, @gp, -100, @infFITC, meanfunc, covfuncF, likfunc, x, y);
[m s2] = gp(hyp, @infFITC, meanfunc, covfuncF, likfunc, x, y, z);

% Take a threshold to assign to one of the class
yhatGP = outputLabelsFromPrediction(m, 0);


%% GP large scale Classification

x = Tr.X(1:500,:);
y = Tr.y(1:500);
t = Te.X(1:500,:);
n = size(x,1);


% Mean function
meanfunc = @meanConst; 
hyp.mean = 0;

% use inducing points u and to base the computations on cross-covariances 
% between training, test and inducing points only
nu = fix(n); iu = randperm(n); iu = iu(1:nu); u = x(iu,:);

% Covariance function
covfunc = @covSEiso; 
ell = 1.0; % characteristic length-scale
sf = 1.0; % standard deviation of the signal sf
hyp.cov = log([ell sf]);
covfuncF = {@covFITC, {covfunc}, u};

% Likelihood function
likfunc = @likErf;

inffunc = @infFITC_EP;                       % also @infFITC_Laplace is possible

hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfuncF, likfunc, x, y);
[a, b, c, d, lp] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, x, y, t, ones(n,1));

% Note: Output arguments: 
% When computing test probabilities, we call gp with additional test inputs, 
% and as the last argument a vector of targets for which the log probabilities 
% lp should be computed. The fist four output arguments of the function are mean 
% and variance for the targets and corresponding latent variables respectively.

% lp gives the logarithm probabilities. Get the exp to have predictions
gpPred = exp(lp);
yHatGP = outputLabelsFromPrediction(gpPred, 0.5);

% We do recover our large rate of negative VS positive images
hist(yHatGP);

res = fastROC(Te.y(1:500) > 0, gpPred);

% TODO : Play with parameters + ROC Curve

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