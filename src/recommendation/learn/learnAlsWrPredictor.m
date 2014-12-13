function predictor = learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda, displayLearningCurve)
% LEARNALSWRPREDICTOR This predictor leverages to low-rank approximation
% computed with ALS-WR.
%
% INPUT
%   nFeatures: target reduced dimensionality (number of features to preserve)
%   lambda:    regularization term
%   displayLearningCurve: boolean flag to show the learning curve
%                         of the ALS-WR algorithm

    % TODO: auto-learn lambda

    [U, M] = alswr(Y, Ytest, nFeatures, lambda, displayLearningCurve);
    
    % Biais correction
    % TODO: do we really have the right to use the test set's mean
    % for post-processing?
    [idx, sz] = getRelevantIndices(Ytest);
    Yhat = reconstructFromLowRank(U, M, idx, sz);
    biais = mean(nonzeros(Yhat)) - mean(nonzeros(Ytest));
    
    predictor = @(user, artist) U(:, user)' * M(:, artist) - biais;
end
