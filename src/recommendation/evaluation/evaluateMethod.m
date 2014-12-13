function [trError, teError] = evaluateMethod(getPredictor, Ytrain, Ytest, userDV, artistDV)
% EVALUATEMETHOD
%
% INPUT
%   getPredictor: function(Y, Ytest)
%                 which returns a `predict(user, artist)` function
%   Y
%   Ytest
%   userDV:       precomputed derived variables
%   artistDV:     precomputed derived variables
% OUTPUT
%   trError, teError: estimated train and test errors

    % TODO: run on mutliple random seeds
    % TODO: need more input?
    predictor = getPredictor(Ytrain, Ytest, userDV, artistDV);

    % Predict training set
    [trIdx, trSz] = getRelevantIndices(Ytrain);
    trYhat = predictCounts(predictor, trIdx, trSz);
    trError = computeRmse(Ytrain, trYhat);

    % Predict test set
    [teIdx, teSz] = getRelevantIndices(Ytest);
    teYhat = predictCounts(predictor, teIdx, teSz);
    teError = computeRmse(Ytest, teYhat);


    % fprintf('RMSE for method TODO: %f | %f\n', trError, teError);
    diagnoseError(Ytrain, trYhat, Ytest, teYhat);
end
