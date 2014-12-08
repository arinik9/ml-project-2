clearvars;

% add path to source files and toolboxs
%addpath(genpath('./toolbox/DeepLearnToolbox'));
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

[coeff,score,latent,tsquared,explained] = pca(Tr.normX);

% Plot on 1st and 2nd principal component
figure()
plot(score(:,1),score(:,2),'+')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')

% how to get 95% of representation as it should ?
figure()
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')

expl = cumsum(explained);

fprintf('Done! You''re now in a lower dimension space !\n');

%% Plot PCA Representation

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

%[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

Tr.tXpca = [ones(size(Tr.Xpca,1),1), Tr.Xpca];
%Te.tX = [ones(size(Te.normX,1),1), Te.normX];

alpha = 0.5;
lambda = 100;

beta = penLogisticRegressionAuto(Tr.y, Tr.Xpca);
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
% TO DO : Apply PCA and classify after that

% Note: if the dimensionality is too big (very fat x matrix) we only get
% negative labels... -> PCA + stochastic ?
x = Tr.X(1:1200,1:500);
y = Tr.y(1:1200);
t = Te.X(:,1:500);
n = size(t,1);

% n1 = 80; n2 = 40;                   % number of data points from each class
% S1 = eye(2); S2 = [1 0.95; 0.95 1];           % the two covariance matrices
% m1 = [0.75; 0]; m2 = [-0.75; 0];                            % the two means
%  
% x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);
% x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);
%  
% xB = [x1 x2]'; yB = [-ones(1,n1) ones(1,n2)]';
% 
% [t1, t2] = meshgrid(-4:0.1:4,-4:0.1:4);
% tB = [t1(:) t2(:)]; nB = length(t); 

% [u1,u2] = meshgrid(linspace(-2,2,5)); u = [u1(:),u2(:)]; clear u1; clear u2
% nu = size(u,1);
% covfuncF = {@covFITC, {covfunc}, u};
% inffunc = @infFITC_EP;                       % also @infFITC_Laplace is possible
% hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfuncF, likfunc, x, y);
% [a b c d lp] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, x, y, t, ones(n,1));
% 
% gpPred = exp(lp);

gpPred = GPClassificationPrediction(y,x,t);

% meanfunc = @meanConst; hyp.mean = 0;
% covfunc = @covSEiso; ell = 1.0; sf = 1.0; hyp.cov = log([ell sf]);
% likfunc = @likErf;
% 
% [u1,u2] = meshgrid(linspace(-2,2,5)); u = [u1(:),u2(:)]; clear u1; clear u2
% nu = size(u,1);
% covfuncF = {@covFITC, {covfunc}, u};
% inffunc = @infFITC_EP; 
% 
% hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfuncF, likfunc, x, y);
% [a b c d lp] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, x, y, t, ones(n, 1));
% gpPred = exp(lp)

yHatGP = outputLabelsFromPrediction(gpPred, 0.5);

% We do recover our large rate of negative VS positive images
hist(yHatGP);

%% Test GP
% Methods names for legend
methodNames = {'GP Classification','Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( Te.y(1:500) > 0, [gpPred, randPred(1:500)], true, methodNames );

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