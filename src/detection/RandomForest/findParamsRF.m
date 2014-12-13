function [nTreesStar, nMinLeafStar, trainTPR, testTPR] = findParamsRF(y, X, k, nTreesValues, minLeafValues,seed)

    nDf = length(nTreesValues);
    nLeafs = length(minLeafValues);

    trainTPR = zeros(nDf,1);
    testTPR = zeros(nDf,1);
    bestTPR = 0;
    nTreesStar = 0;
    nMinLeafStar = 0;
    
    for i=1:nDf
        
        nTrees = nTreesValues(i);
        
        for l=1:nLeafs
            
            leaf = minLeafValues(l);
        
            learn = @(y, X) trainRandomForest(y, X, nTrees, leaf);
            predict = @(model, X) predictRandomForest(model, X);
            computePerformances = @(trueOutputs, pred, model_name) kCVfastROC(trueOutputs, pred, model_name, 0);

            setSeed(seed);
            [trainTPR(i,l), testTPR(i,l)] = kFoldCrossValidation(y, X, k, learn, predict, computePerformances);

            if (testTPR(i,l) > bestTPR)
                nTreesStar = nTrees;
                nMinLeafStar = leaf;
                bestTPR = testTPR(i,l);
            end

            % Status
            fprintf('TPR for nTrees = %f nMinLeaf = %f : train %f | test %f\n', nTrees, leaf, trainTPR(i,l), testTPR(i,l));

        end
        
    end
    
    % Plot evolution of train and test performances with respect to nTree
    % and nMinLeafs
    styles = {'r','b','k','m','g','y','r--', 'b--', 'k--','m--','g--','y--'};
    
    for l=1:nLeafs
        legendNames{l} = sprintf('minLeaf = %d', minLeafValues(l));
    end
    
    figure(1);
    for l=1:nLeafs
        plot(nTreesValues, trainTPR(:,l), styles{l});
        hold on;
    end;
    xlabel('number of Trees');
    ylabel('Training TPR');
    title('Learning curves on train with different MinLeaf values');
    legend( legendNames, 'Location', 'NorthWest' );
    %savePlot('./report/figures/detection/randomforest-findparams-train.pdf');

    figure(2);
    for l=1:nLeafs
        plot(nTreesValues, testTPR(:,l), styles{l});
        hold on;
    end;
    xlabel('number of Trees');
    ylabel('Training (blue) and test (red) TPR');
    title('Learning curves on test with different MinLeaf values');
    legend( legendNames, 'Location', 'NorthWest' );
    %savePlot('./report/figures/detection/randomforest-findparams-test.pdf');
    
end