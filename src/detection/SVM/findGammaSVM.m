function [bestGamma, testTPR, trainTPR] = findGammaSVM(y, X, k, gammaValues, seed)
% Finds the best gamma parameter for RBF kernel SVM from a given range of
% values gammaValues over k-fold CV and plots learning curves on train and test data

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
        computePerformances = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 0, 0, model_name);
        
        rng('default');
        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances, 0);
         
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