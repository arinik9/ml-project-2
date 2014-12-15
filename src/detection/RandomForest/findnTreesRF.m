function [nTreesStar, trainTPR, testTPR] = findnTreesRF(y, X, k, nTreesValues, seed)
% Finds the best number of trees for random forests from a given range of values 
% nTreesValues over k-fold CV and plots learning curves on train and test data

    if (nargin < 5)
        seed = 1;
    end

    nDf = length(nTreesValues);

    trainTPR = zeros(nDf,1);
    testTPR = zeros(nDf,1);
    bestTPR = 0;
    nTreesStar = 0;
    
    for i=1:nDf
        
        nTrees = nTreesValues(i);
        
        learn = @(y, X) trainRandomForest(y, X, nTrees);
        predict = @(model, X) predictRandomForest(model, X);
        computePerformances = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 0, 0, model_name);
        
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances, 0);
  
        if (testTPR(i) > bestTPR)
            nTreesStar = nTrees;
            bestTPR = testTPR(i);
        end
        
        % Status
        fprintf('TPR for nTrees = %f: train %f | test %f\n', nTrees, trainTPR(i), testTPR(i));
        
    end
    
    % Plot evolution of train and test error with respect to lambda
    figure;
    plot(nTreesValues, trainTPR, '.-b');
    hold on;
    plot(nTreesValues, testTPR, '.-r');
    xlabel('number of Trees');
    ylabel('Training (blue) and test (red) TPR');
    
end