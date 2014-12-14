function predictor = learnGMMPredictor(Ytrain, Ytest, ~, ~, reduceSpace, K)
% LEARNGMMPREDICTOR Predictions based on Gaussian Mixture Model clustering
% (soft clustering) computed via the Variational Bayesian
% Inference for Gaussian mixture algorith (`vbgm.m`)
% from Michael Chen
%
% Since clustering on 15082-dimensional data is too expensive with this algorithm,
% we act on a low-rank approximation computed with the function `reduceSpace`.
%
% Prediction is then done from the ALS reconstruction, using the soft
% cluster assignment as weights
%
% INPUT
%   reduceSpace: function(Ytrain, Ytest) returning a low-rank
%                approximation of Ytrain
%   K: maximum number of clusters (the most appropriate
%      number is selected by the algorithm)
%   S: precomputed similarity matrix
%
% SEE ALSO
%   http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model

    [U, M] = reduceSpace(Ytrain, Ytest);
    [idx, sz] = getRelevantIndices(Ytrain);
    Yhat = reconstructFromLowRank(U, M, idx, sz);

    biais = mean(nonzeros(Yhat)) - mean(nonzeros(Ytrain));

    fprintf('Starting Gaussian Mixture Model clustering (%d clusters at most)...\n', K);
    % model.m gives the clusters' centroids
    % model.R gives the soft assignments
    [~, model, ~] = vbgm(U, K);
    fprintf('Done!\n');
    
    components = size(model.m, 1);
    
    predictor = @(user, artist) ...
        sum(repmat(model.R(user, :), [components, 1]) .* model.m, 2)' ...
        * M(:, artist) ...
        - biais;
end
