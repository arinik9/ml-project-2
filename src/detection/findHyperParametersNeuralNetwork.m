function [dropOutStar, weightDecayStar, trainTPR, testTPR] = findHyperParametersNeuralNetwork(y, X, k, dropOutFractions, weightDecays,seed)

    nDf = length(dropOutFractions);
    nWd = length(weightDecays);

    trainTPR = zeros(nDf,1);
    testTPR = zeros(nDf,1);
    bestTPR = 0;
    dropOutStar = 0;
    weightDecayStar = 0;
    
    for d=1:nDf
        
        dpFraction = dropOutFractions(d);
        
        for w=1:nWd
            
            weightDecay = weightDecays(w);

            learn = @(y, X) learnNeuralNetwork(y, X, 0, 1, 'sigm', dpFraction, weightDecay);
            predict = @(model, X) predictNeuralNetwork(model, X);

            setSeed(seed);
            [trainTPR(d), testTPR(d)] = kFoldCrossValidation(y, X, k, learn, predict, 0);

            if (testTPR(d) > bestTPR)
                dropOutStar = dpFraction;
                weightDecayStar = weightDecay;
                bestTPR = testTPR(d);
            end

            % Status
            fprintf('TPR for dropout = %f weight decay = %f : train %f | test %f\n', dpFraction, weightDecay, trainTPR(d), testTPR(d));
        end
    end
    
    % Plot evolution of train and test error with respect to lambda
    %{
    figure;
    semilogx(dropOutFractions, trainTPR, '.-b');
    hold on;
    semilogx(dropOutFractions, testTPR, '.-r');
    xlabel('Drop out');
    ylabel('Training (blue) and test (red) TPR');
    %}
    
end