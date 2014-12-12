function [trAvgTPR, teAvgTPR, predTr, predTe, trueTr, trueTe] = kFoldCrossValidation(y, X, K, learnModel, predict, computePerformance, plot_flag, model_name)
% Perform k-fold cross validation to obtain an estimate of the train and
% test performances. It also returns matrices of all predictions of test
% and train splits as well as corresponding true labels to be used to plot
% multiple ROC Curves on a same figure.
% Inputs:
%   - y: output features
%   - X: input features
%   - K: number of folds
%   - learnModel: function(y, X) to obtain the model parameters
%   - predict: function(model, X) to compute the prediction from the input and
%     model parameters
%   - plot_flag: set to 1 to plot obtained average curve
%   - model_name: name of the applied model used in the plot title
% Outputs:
%   - trAvgTPR: average true positive rate on all train splits (train
%   performance estimate)
%   - teAvgTPR: average true positive rate on all test splits (test
%   performance estimate)

% TODO: kCVfastROC outside so that kCV is plot_flag, model_name etc free?

    if (~exist('plot_flag','var') && plot_flag ~= 0 && plot_flat ~= 1 )
        plot_flag = 0;
    end
    
    if ~exist('model_name','var')
        model_name = 'Model applied';
    end

    % Split data in k folds (create indices only)
    N = size(y, 1);
    idx = randperm(N);
    Nk = floor(N / K);
    cvIndices = zeros(K, Nk);
    for k = 1:K
        cvIndices(k, :) = idx( (1 + (k-1)*Nk):(k * Nk) );
    end;
    
    %subTrAvgTPR = zeros(K, 1);
    %subTeAvgTPR = subTrAvgTPR;
    
    % For each fold, compute the train and test predictions with the learnt model
    for k = 1:K
        % Get k'th subgroup in test, others in train
        idxTe = cvIndices(k, :);
        idxTr = cvIndices([1:k-1 k+1:end], :);
        idxTr = reshape(idxTr, numel(idxTr), 1);
        yTe = y(idxTe);
        XTe = X(idxTe, :);
        yTr = y(idxTr);
        XTr = X(idxTr, :);

        % Store true labels for ROC Curve
        trueTr(:,k) = yTr;
        trueTe(:,k) = yTe;

        % Learn model parameters
        model = learnModel(yTr, XTr);
        
        % Make predictions on test and train
        predTr(:,k) = predict(model, XTr);
        predTe(:,k) = predict(model, XTe);
        
        % Regular sub error computation for each split but handled below by
        % kCVfastROC
        %subTrAvgTPR(k) = fastROC(yTr > 0, predTr(:,k),0)
        %subTeAvgTPR(k) = fastROC(yTe > 0, predTe(:,k),0); 
        
        %fprintf('avgTPR on train : %d | avgTPR on test : %d \n', subTrAvgTPR(k), subTeAvgTPR(k));
       
    end;
    
    % Compute training and test error for every k'th train / test split.
    % Returns a vector of average TPR computed for each k train / test
    % split
    
    if (plot_flag == 1)
        figure;
    end
    avgTPRtrain = computePerformance(trueTr, predTr, plot_flag, strcat(model_name, ' on train data'));
    
    if (plot_flag == 1)
        figure;
    end
    avgTPRtest = computePerformance(trueTe, predTe, plot_flag, strcat(model_name, ' on test data'));

    % Estimate test and train performances are the average over k folds
    trAvgTPR = mean(avgTPRtrain);
    teAvgTPR = mean(avgTPRtest);
end