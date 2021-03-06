addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));
clearvars;

loadDataset;
setSeed(1);
[~, Ytest, ~, Ytrain, ~] = getTrainTestSplit(Yoriginal, Goriginal, 0.3, 0, 3);
[userDV, artistDV] = generateDerivedVariables(Ytest);

%% Testing VBGM toolbox
% From http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model

% Generate dummy Gaussian clusters
numClusters = 3;
nPerCluster = 200;
dummyData = [];

sigmaX = 3; 
sigmaY = 5;
for ii = 1:numClusters 
    sigmas = diag([sigmaX sigmaY]); 
    centers = diag([100 -200]);

    simGauss = sigmas * randn(2, nPerCluster); 
    mu = centers * rand(2, 1); 
    cluster = bsxfun(@plus, simGauss, mu); 

    dummyData = cat(2, dummyData, cluster); 
end;

[labels, model, L] = vbgm(dummyData, 5);

figure;
plot(dummyData(1, :), dummyData(2, :), '.');
hold on;

plot(model.m(1, :)', model.m(2, :)', '.');

% labels gives the hard assignments
% model.m gives the clusters' centroids
% model.R gives the soft assignments

%% Gaussian Mixture Model (soft clustering)
lambda = 0.5;
nComponents = [50 100 200 500];
nClusters = [30 50 100 500];

e = {};
e.tr = zeros(length(nComponents), length(nClusters));
e.te = zeros(length(nComponents), length(nClusters));

% Select hyperparameters values
for i = 1:length(nComponents)
    components = nComponents(i);
    [U, M] = alswr(Ytrain, Ytest, components, lambda, 1, 5);
    
    for j = 1:length(nClusters)
        maxClusters = nClusters(j);
        [~, model, ~] = vbgm(U, maxClusters);

        % TODO: predict something more interesting than the ALS reconstruction
        predictor = @(user, artist) ...
            sum(repmat(model.R(user, :), [components, 1]) .* model.m, 2)' ...
            * M(:, artist);

        [idx, sz] = getRelevantIndices(Ytrain);
        Yhat = predictCounts(predictor, idx, sz);
        e.tr(i, j) = computeRmse(Ytrain, Yhat);
        
        [idx, sz] = getRelevantIndices(Ytest);
        Yhat = predictCounts(predictor, idx, sz);
        e.te(i, j) = computeRmse(Ytest, Yhat);
        fprintf('%d components, %d clusters: %f | %f\n', components, maxClusters, e.tr(i, j), e.te(i, j));
        %diagnoseError(Ytrain, Yhat);
    end;
end;

%% Trying to cluster directly on the dataset
% Probably not a good idea
%{
topArtists = find(artistDV(:, 2) > 10);
Ysmall = Ytrain(:, topArtists);
% Artist-oriented clustering
[labels, model, L] = vbgm(Ysmall, 30);
%%
N = size(Ysmall, 2);
predictor = @(user, artist) ...
    sum(model.R(artist, :) .* model.m(user, :), 2)';

[idx, sz] = getRelevantIndices(Ysmall);
Yhat = predictCounts(predictor, idx, sz);
rmse = computeRmse(Ysmall, Yhat);
rmse
diagnoseError(Ysmall, Yhat);
%}
