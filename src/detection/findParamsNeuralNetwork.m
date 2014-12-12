function [dropOutStar, weightDecayStar, trainTPR, testTPR] = findParamsNeuralNetwork(y, X, k, dropOutFractions, weightDecays,seed)

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
            [trainTPR(d,w), testTPR(d,w)] = kFoldCrossValidation(y, X, k, learn, predict, 0);

            if (testTPR(d,w) > bestTPR)
                dropOutStar = dpFraction;
                weightDecayStar = weightDecay;
                bestTPR = testTPR(d,w);
            end

            % Status
            fprintf('TPR for dropout = %f weight decay = %f : train %f | test %f\n', dpFraction, weightDecay, trainTPR(d), testTPR(d));
        end
    end
    
        % Plot evolution of train and test error with respect to lambda
    styles = {'r','b','k','m','g','y','r--', 'b--', 'k--','m--','g--','y--'};
    
    for n=1:nLeafs
        legendNames{n} = sprintf('weightDecay = %d', weightDecays(n));
    end
    
    figure(1);
    for n=1:nWd
        semilogx(dropOutFractions, trainTPR(:,n), styles{n});
        hold on;
    end;
    xlabel('drop out fraction');
    ylabel('Training TPR');
    title('Learning curves on train with different MinLeaf values');
    legend( legendNames, 'Location', 'NorthWest' );
    
    figure(2);
    for n=1:nWd
        semilogx(dropOutFractions, testTPR(:,n), styles{n});
        hold on;
    end;
    xlabel('drop out fraction');
    ylabel('Training (blue) and test (red) TPR');
    title('Learning curves on test with different Weight Decay values');
    legend( legendNames, 'Location', 'NorthWest' );
    
end