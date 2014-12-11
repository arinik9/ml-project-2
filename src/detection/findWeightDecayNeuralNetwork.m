function [weightDecayStar, trainTPR, testTPR] = findWeightDecayNeuralNetwork(y, X, k, weightDecaysValues, seed)

    n = length(weightDecaysValues);

    trainTPR = zeros(n,1);
    testTPR = zeros(n,1);
    bestTPR = 0;
    weightDecayStar = 0;
    
    for i=1:n
        
        weightDecay = weightDecaysValues(i);
        
        learn = @(y, X) learnNeuralNetwork(y, X, 0, 1, 'sigm', 0, weightDecay);
        predict = @(model, X) predictNeuralNetwork(model, X);
        
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, 0);
        
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