function [bestParam, trainTPR, testTPR] = find1Param(y, X, k, trainModel, predictModel, paramValues, seed)
%Not working. kCV returns same test and train error for every paramValue
    if (nargin < 5)
        seed = 1;
    end

    n = length(paramValues);

    trainTPR = zeros(n,1);
    testTPR = zeros(n,1);
    bestTPR = -1;
    bestParam = -1;

    for i=1:n

        p = paramValues(i);

        train = @(y, X) trainModel(y, X, p);
        predict = @(model, X) predictModel(model, X);
        computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 0, 0, model_name);

        setSeed(seed);
        [trainTPR(i), testTPR(i)] = kFoldCrossValidation(y, X, k, train, predict, computePerformance, 0, '');

        if (testTPR(i) > bestTPR)
            bestParam = p;
            bestTPR = testTPR(i)
        end

        % Status
        fprintf('TPR for param = %f: train %f | test %f\n', p, trainTPR(i), testTPR(i));

    end

    % Plot evolution of train and test error with respect to the parameter values
    figure;
    plot(paramValues, trainTPR, '.-b');
    hold on;
    plot(paramValues, testTPR, '.-r');
    xlabel('Range Values');
    ylabel('Training (blue) and test (red) TPR');

end
