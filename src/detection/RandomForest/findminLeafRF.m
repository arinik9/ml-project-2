function [nMinLeafStar, trainTPR, testTPR] = findminLeafRF(y, X, k, leafValues, seed)

    nDf = length(leafValues);

    trainTPR = zeros(nDf,1);
    testTPR = zeros(nDf,1);
    bestTPR = 0;
    nMinLeafStar = 0;
    
    for i=1:nDf
        
        nLeaf = leafValues(i);
        
        learn = @(y, X) trainRandomForest(y, X, 100, sqrt(size(X,2)), nLeaf);
        predict = @(model, X) predictRandomForest(model, X);
        computePerformances = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 0, 0, model_name);
        
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances, 0);
  
        if (testTPR(i) > bestTPR)
            nMinLeafStar = nLeaf;
            bestTPR = testTPR(i);
        end
        
        % Status
        fprintf('TPR for nTrees = %f: train %f | test %f\n', nLeaf, trainTPR(i), testTPR(i));
        
    end
    
    % Plot evolution of train and test error with respect to lambda
    figure;
    plot(leafValues, trainTPR, '.-b');
    hold on;
    plot(leafValues, testTPR, '.-r');
    xlabel('number of Trees');
    ylabel('Training (blue) and test (red) TPR');
    
end