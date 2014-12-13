function [ellStar, sfStar, trainTPR, testTPR] = findCovHypGP(y, X, k, ellValues, sfValues, seed)

    if (nargin < 6)
        seed = 1;
    end

    nEll = length(ellValues);
    nSf = length(sfValues);

    trainTPR = zeros(nEll,1);
    testTPR = zeros(nEll,1);
    bestTPR = 0;
    ellStar = 0;
    sfStar = 0;
    
    for d=1:nEll
        
        ell = ellValues(d);
        
        for w=1:nSf
            
            sf = sfValues(w);

            learn = @(y, X) trainGPClassification(y, X, @likLogistic, @infFITC_Laplace, @covSEiso, log([ell sf]));
            predict = @(model, X) predictGPClassification(model, X);
            computePerformances = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 0, 0, model_name);
       
            setSeed(seed);
            [trainTPR(d,w), testTPR(d,w)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances, 0);

            if (testTPR(d,w) > bestTPR)
                ellStar = ell;
                sfStar = sf;
                bestTPR = testTPR(d,w);
            end

            % Status
            fprintf('TPR for dropout = %f weight decay = %f : train %f | test %f\n', ell, sf, trainTPR(d), testTPR(d));
        end
    end
    
        % Plot evolution of train and test error with respect to lambda
    styles = {'r','b','k','m','g','y','r--', 'b--', 'k--','m--','g--','y--'};
    
    for n=1:nSf
        legendNames{n} = sprintf('weightDecay = %d', sfValues(n));
    end
    
    figure(1);
    for n=1:nSf
        plot(ellValues, trainTPR(:,n), styles{n});
        hold on;
    end;
    xlabel('Length scale values');
    ylabel('Training TPR');
    title('Learning curves on train with different signal magnitude');
    legend( legendNames, 'Location', 'NorthWest' );
    
    figure(2);
    for n=1:nSf
        plot(ellValues, testTPR(:,n), styles{n});
        hold on;
    end;
    xlabel('Length scale values');
    ylabel('Training (blue) and test (red) TPR');
    title('Learning curves on test with different signal magnitude');
    legend( legendNames, 'Location', 'NorthWest' );
    
end