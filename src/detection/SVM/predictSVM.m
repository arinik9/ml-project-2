function [svmPred, labels, accuracy] = predictSVM(svmModel, Xpred, ypred)
% Takes a SVM model and computes predictions for the given Xpred inputs
% INPUTS
%   svmModel:     trained SVM model using trainSVM (or svmtrain method from libsvm)
%   Xpred:        input test data
%   ypred:        outputs test data (if unknown, put random values according to
%   			  the lib)
% OUTPUTS
%   svmPred:      probabilities predictions
%   labels:       default labels predictions made by the SVM

    if (nargin < 3)
       ypred = ones(size(Xpred,1),1); 
    end

    [labels, accuracy, prob_estimates] = svmpredict(ypred, Xpred, svmModel, '-b 1');
    
    % get scores (probability of predicting label +1)
    svmPred  = prob_estimates(:, 1);

end