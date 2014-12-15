function [nActivationStar, trainTPR, testTPR] = findActivationsNeuralNetwork(y, X, k, activationValues, seed)
% Finds the best activation functions number on L2 and for NN from a given range of values 
% activationValues over k-fold CV and plots learning curves on train and test data

    if (nargin < 5)
        seed = 1;
    end

    nDf = length(activationValues);

    trainTPR = zeros(nDf,1);
    testTPR = zeros(nDf,1);
    bestTPR = 0;
    nActivationStar = 0;
    
    for i=1:nDf
        
        a = activationValues(i);
        
        learn = @(y, X) trainNeuralNetwork(y, X, 0, 1, 'sigm', 0, 0, [size(X,2) a 2]);
        predict = @(model, X) predictNeuralNetwork(model, X);
        computePerformances = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 0, 0, model_name);
        
        rng('default');
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances, 0);
        
        if (testTPR(i) > bestTPR)
            nActivationStar = a;
            bestTPR = testTPR(i);
        end
        
        % Status
        fprintf('TPR for # activation = %f: train %f | test %f\n', a, trainTPR(i), testTPR(i));
        
    end
    
    % Plot evolution of train and test error with respect to lambda
    figure;
    plot(activationValues, trainTPR, '.-b');
    hold on;
    plot(activationValues, testTPR, '.-r');
    xlabel('Activation functions number on L2');
    ylabel('Training (blue) and test (red) TPR');
    
end