function [trAvgTPR, teAvgTPR] = kFoldCrossValidation(y, tX, K, learnModel, predict)
% Perform k-fold cross validation to obtain an estimate of the test error
% K: number of folds
% learnModel: function(y, tX) to obtain the model parameters
% predict: function(model, tX) to compute the prediction from the input and
%          model parameters
% computeError: function(y, yHat) to compute the error about the prediction

    % Split data in k folds (create indices only)
    N = size(y, 1);
    idx = randperm(N);
    Nk = floor(N / K);
    cvIndices = zeros(K, Nk);
    for k = 1:K
        cvIndices(k, :) = idx( (1 + (k-1)*Nk):(k * Nk) );
    end;
    
    % For each fold, compute the train and test error with the learnt model
    subTrError = zeros(K, 1);
    subTeError = subTrError;
    for k = 1:K
        % Get k'th subgroup in test, others in train
        idxTe = cvIndices(k, :);
        idxTr = cvIndices([1:k-1 k+1:end], :);
        idxTr = reshape(idxTr, numel(idxTr), 1);
        yTe = y(idxTe);
        XTe = tX(idxTe, :);
        yTr = y(idxTr);
        XTr = tX(idxTr, :);

        % Learn model parameters
        model = learnModel(yTr, XTr);
        
        % Make predictions on test and train
        predTr = predict(model, XTr);
        predTe = predict(model, XTe);
        
        % Compute training and test error for k'th train / test split
        subTrAvgTPR(k) = fastROC(yTr > 0, predTr); 
        subTeAvgTPR(k) = fastROC(yTe > 0, predTe); 
        fprintf('avgTPR on train : %d | avgTPR on test : %d \n', subTrAvgTPR(k), subTeAvgTPR(k));
        
        
    end;

    % Estimate test and train errors are the average over k folds
    trAvgTPR = mean(subTrAvgTPR);
    teAvgTPR = mean(subTeAvgTPR);
end