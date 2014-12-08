function [train_scores, test_scores] = prinCompProjection(coeff, Xtrain, Xtest, nPrinComp)
% Projects train and test data in the reduced space explained by the
% selected principal components
% Input:
%   - coeff: obtained thanks to pca() function. Each olumn containing 
%     coefficients for one principal component. The columns are in order 
%     of decreasing component variance.
%   - Xtrain: normalized train examples
%   - Xtest: normalized test examples
%   - nPrinComp: number of principal components kept
% Output:
%   - train_scores: train data projected in reduced space
%   - test_scores: test data projected in reduced space

    % Get the principal component we are projecting on
    pc = coeff(:,1:nPrinComp);

    % Compute train and test scores 
    train_scores = Xtrain * pc;
    % Equivalent to previous line but need to be given PCA score as parameter
    % train_scores = score(:,1:nPrinComp);

    test_scores = Xtest * pc;

end