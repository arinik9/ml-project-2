function [bestNvarToSample, trainTPR, testTPR] = findnVarSampleRF(y, X, k, rangeValues, seed)

    if (nargin < 6)
        seed = 1;
    end

    n = length(rangeValues);

    trainTPR = zeros(n,1);
    testTPR = zeros(n,1);
    bestTPR = 0;
    bestNvarToSample = 0;
    
    for i=1:n
        
        varNumber = rangeValues(i);
        
        learn = @(y, X) trainRandomForest(y, X, 100, varNumber, 1);
        predict = @(model, X) predictRandomForest(model, X);
        computePerformances = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 0, 0, model_name);
        
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances, 0);
        
        if (testTPR(i) > bestTPR)
            bestNvarToSample = varNumber;
            bestTPR = testTPR(i);
        end
        
        % Status
        fprintf('TPR for # variables = %f: train %f | test %f\n', varNumber, trainTPR(i), testTPR(i));
        
    end
    
    % Plot evolution of train and test error with respect to lambda
    figure;
    plot(rangeValues, trainTPR, '.-b');
    hold on;
    plot(rangeValues, testTPR, '.-r');
    xlabel('Number of variables to sample');
    ylabel('Training (blue) and test (red) TPR');
    
end