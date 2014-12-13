function beta = ridgeRegression(y, tX, lambda)
% Learn model parameters beta using Ridge Regression
% lambda: penalization parameter
    gramMatrix = (tX' * tX);
    penalizationTerm = eye(size(gramMatrix));
    % preventing from lifting beta0 value
    penalizationTerm(:,1) = zeros(size(penalizationTerm, 1), 1);
    l = lambda * penalizationTerm;
    beta = (gramMatrix + l) \ (tX' * y);
end
