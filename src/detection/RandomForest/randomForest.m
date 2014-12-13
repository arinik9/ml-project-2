%getStartedDetection;

% Split data into train and test set given a proportion
prop = 2/3;
fprintf('Splitting into train/test with proportion %.2f..\n', prop);
[Tr.pcaX, Tr.y, Te.pcaX, Te.y] = splitDataDetection(y, pcaX, prop);

%% Model

% Parameters to play with:
% - nTrees: Number of trees in the random forest
% - 'NVarToSample': Number of variables to select at random for each decision split. 
%                 Default is the square root of the number of variables for classification
% - 'MinLeaf': Minimum number of observations per tree leaf. Default is 1 for classification
% - 'FBoot': Fraction of input data to sample with replacement from the input data for growing each new tree.
%            set 'SampleWithReplacement' to off to play with it

%% Number of trees in the forest
nTreesValues = [50 100 200 300 400 500];
[bestnTree, trainTPR, testTPR] = findnTreesRF(y, pcaX, 3, nTreesValues, 1);

% TODO: Try to run it on non-PCA values
% Clearly overfitting on PCA train data with just nTreesValues used Best
% values for 400 Trees, but 100 gives good results as well.

%% kCV Parameters validation

nTreesValues = [50 100];
nMinLeafs = [1 10];
[bestnTree, bestMinLeaf, trainTPR, testTPR] = findParamsRF(y(1:500), pcaX(1:500,:), 3, nTreesValues, nMinLeafs, 1);

%% Random prediction
randPred = rand(size(Te.y)); 

%% Evaluate

% Methods names for legend
methodNames = {'Random Forest','Random'};

% Prediction performances on different models
avgTPRList = evaluateMultipleMethods( Te.y > 0, [rfPred, randPred], true, methodNames );

