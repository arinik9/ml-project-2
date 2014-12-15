function [nMinLeafStar, trainTPR, testTPR] = findminLeafRF(y, X, k, leafValues, seed)
% Finds the best number of minimum observation per leaf for random forests from a given range of values 
% leafValues over k-fold CV and plots learning curves on train and test data

    if (nargin < 5)
        seed = 1;
    end

    nDf = length(leafValues);

    trainTPR = zeros(nDf,1);
    testTPR = zeros(nDf,1);
    bestTPR = 0;
    nMinLeafStar = 0;
    
    for i=1:nDf
        
        nLeaf = leafValues(i);
        
        learn = @(y, X) trainRandomForest(y, X, 100, sqrt(size(X,2)) / 2, nLeaf);
        predict = @(model, X) predictRandomForest(model, X);
        computePerformances = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 0, 0, model_name);
        
        rng('default');
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances, 0);
  
        if (testTPR(i) > bestTPR)
            nMinLeafStar = nLeaf;
            bestTPR = testTPR(i);
        end
        
        % Status
        fprintf('TPR for minLeaf = %f: train %f | test %f\n', nLeaf, trainTPR(i), testTPR(i));
        
    end
    
    % Plot evolution of train and test error with respect to lambda
    figure;
    plot(leafValues, trainTPR, '.-b');
    hold on;
    plot(leafValues, testTPR, '.-r');
    xlabel('number of minLeaf');
    ylabel('Training (blue) and test (red) TPR');
    
end