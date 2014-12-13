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
    predictor = @(user, artist) U(:, user)' * M(:, artist);
end
