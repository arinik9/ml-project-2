function err = computeLogisticRegressionMse(y, tX, beta)
% Alternative form of computeLRMse (kept for compatibility purpose)
    err = computeLRMse(y, tX * beta);
end