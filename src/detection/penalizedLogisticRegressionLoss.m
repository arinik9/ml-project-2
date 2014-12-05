function [err, gradient, hessian] = penalizedLogisticRegressionLoss(y, tX, beta, lambda)
    [err, gradient, hessian] = logisticRegressionLoss(y, tX, beta);
    
    % Never penalize beta0
    lBeta = lambda * beta;
    lBeta(1) = 0;

    n = size(y, 1);

    err = err + beta' * lBeta / (2 * n);
    gradient = gradient + lBeta / n;
    hessian = hessian + lambda / n;
end

function [err, gradient, hessian] = logisticRegressionLoss(y, tX, beta)
    err = computeLogisticRegressionMse(y, tX, beta);
    gradient = computeLogisticRegressionGradient(y, tX, beta);
    
    sigmoid = exp(logSigmoid(tX * beta));
    S = diag(sigmoid .* (1 - sigmoid));
    hessian = (tX' * S * tX) / size(y, 1);
end

function g = computeLogisticRegressionGradient(y, tX, beta)
% Gradient computation for the Maximum Likelihood Estimator
% of logistic regression
    
	A = tX * beta;
	lSigmoid = logSigmoid(A);
	g = (tX' * (exp(lSigmoid) - y)) / size(y, 1);
end
