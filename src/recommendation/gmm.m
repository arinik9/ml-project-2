addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));
clearvars;

loadDataset;
setSeed(1);
[~, Ytest, ~, Ytrain, ~] = getTrainTestSplit(Yoriginal, Goriginal, 0.3, 0, 3);

K = 30;
lambda = 0.000001;
[U, M] = alswr(Ytrain, Ytest, K, lambda, 1);

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
lambda = 0.000001;
nComponents = [5 50 100 200 500];
nClusters = [3 10 30 50 100 500];

e = {};
e.tr = zeros(length(nComponents), length(nClusters));
e.te = zeros(length(nComponents), length(nClusters));

% Select hyperparameters values
for i = 1:length(nComponents)
    components = nComponents(i);
    [U, M] = alswr(Ytrain, Ytest, components, lambda, 0, 5);
    
    for j = 1:length(nClusters)
        
        maxClusters = nClusters(i);
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

