function [dropOutStar, trainTPR, testTPR] = findDropoutNeuralNetwork(y, X, k, dropOutFractions, seed)

    nDf = length(dropOutFractions);

    trainTPR = zeros(nDf,1);
    testTPR = zeros(nDf,1);
    bestTPR = 0;
    dropOutStar = 0;
    
    for i=1:nDf
        
        dpFraction = dropOutFractions(i);
        
        learn = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', dpFraction, 0);
        predict = @(model, X) predictNeuralNetwork(model, X);
        computePerformances = @(trueOutputs, pred, model_name) kCVfastROC(trueOutputs, pred, model_name, 0);
        
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances);
        
        if (testTPR(i) > bestTPR)
            dropOutStar = dpFraction;
            bestTPR = testTPR(i);
        end
        
        % Status
        fprintf('TPR for dropout = %f: train %f | test %f\n', dpFraction, trainTPR(i), testTPR(i));
        
    end
    
    % Plot evolution of train and test error with respect to lambda
    figure;
    semilogx(dropOutFractions, trainTPR, '.-b');
    hold on;
    semilogx(dropOutFractions, testTPR, '.-r');
    xlabel('Drop out');
    ylabel('Training (blue) and test (red) TPR');
    
end