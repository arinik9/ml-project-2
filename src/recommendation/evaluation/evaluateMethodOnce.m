function [trError, teError, trYhat, teYhat] = ...
    evaluateMethodOnce(getPredictor, Ytrain, Ytest, Gtrain)
% EVALUATEMETHODONCE One run over a given train / test split

    [userDV, artistDV] = generateDerivedVariables(Ytrain);

    % TODO: pass social network G to learning methods
    predictor = getPredictor(Ytrain, Ytest, userDV, artistDV);

    % Predict training set
    [trIdx, trSz] = getRelevantIndices(Ytrain);
    trYhat = predictCounts(predictor, trIdx, trSz);
    trError = computeRmse(Ytrain, trYhat);

    % Predict test set
    [teIdx, teSz] = getRelevantIndices(Ytest);
    teYhat = predictCounts(predictor, teIdx, teSz);
    teError = computeRmse(Ytest, teYhat);
end
