function [bestGamma, testTPR, trainTPR] = findGammaSVM(y, X, k, gammaValues, seed)

    if (nargin < 5)
        seed = 1;
    end
    
    n = length(gammaValues);
    
    trainTPR = zeros(n,1);
    testTPR = zeros(n,1);
    bestTPR = -1;
    bestGamma = 0;

    for i=1:n
        
        gamma = gammaValues(i);
        
        learn = @(y, X) trainSVM(y, X, strcat('-t 2 -b 1 -e 0.01', sprintf(' -g %.4f',gamma)));
        predict = @(model, X) predictSVM(model, X);
        computePerformances = @(trueOutputs, pred, model_name) kCVfastROC(trueOutputs, pred, model_name, 0);
        
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances);
        
        if (testTPR(i) > bestTPR || bestTPR < 0)
            bestGamma = gamma;
            bestTPR = testTPR(i);
        end
        
        % Status
        fprintf('TPR for param = %f: train %f | test %f\n', gamma, trainTPR(i), testTPR(i));
        
    end

    % Plot evolution of train and test error with respect to lambda
    figure;
    plot(gammaValues, trainTPR, '.-b');
    hold on;
    plot(gammaValues, testTPR, '.-r');
    xlabel('Gamma Values');
    ylabel('Training (blue) and test (red) TPR');
    
end