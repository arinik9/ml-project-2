function [nTreesStar, trainTPR, testTPR] = findnTreesRF(y, X, k, nTreesValues, seed)

    nDf = length(nTreesValues);

    trainTPR = zeros(nDf,1);
    testTPR = zeros(nDf,1);
    bestTPR = 0;
    nTreesStar = 0;
    
    for i=1:nDf
        
        nTrees = nTreesValues(i);
        
        learn = @(y, X) trainRandomForest(y, X, nTrees);
        predict = @(model, X) predictRandomForest(model, X);
        computePerformances = @(trueOutputs, pred, model_name) kCVfastROC(trueOutputs, pred, model_name, 0);
        
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances);
        
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