function [weightDecayStar, trainTPR, testTPR] = findWeightDecayNeuralNetwork(y, X, k, weightDecaysValues, seed)
% Finds the best weight decay parameters for NN from given range of 
% weightDecaysValues over k-fold CV and plots learning curves 
% on train and test data

    if (nargin < 5)
        seed = 1;
    end

    n = length(weightDecaysValues);

    trainTPR = zeros(n,1);
    testTPR = zeros(n,1);
    bestTPR = 0;
    weightDecayStar = 0;
    
    for i=1:n
        
        weightDecay = weightDecaysValues(i);
        
        learn = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, weightDecay);
        predict = @(model, X) predictNeuralNetwork(model, X);
        computePerformances = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 0, 0, model_name);
        
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances, 0);
        
        if (testTPR(i) > bestTPR)
            weightDecayStar = weightDecay;
            bestTPR = testTPR(i);
        end
        
        % Status
        fprintf('TPR for dropout = %f: train %f | test %f\n', weightDecay, trainTPR(i), testTPR(i));
        
    end
    
    % Plot evolution of train and test error with respect to lambda
    figure;
    semilogx(weightDecaysValues, trainTPR, '.-b');
    hold on;
    semilogx(weightDecaysValues, testTPR, '.-r');
    xlabel('Weight Decay on L2');
    ylabel('Training (blue) and test (red) TPR');
    
end