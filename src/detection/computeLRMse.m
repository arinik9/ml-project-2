function err = computeLRMse(y, tXTimesBeta)
% Alternative form of compu
% Error is the negative of log-likelihood (normalized by the number of
% data examples).
    n = size(y, 1);

    lSigmoid = logSigmoid(tXTimesBeta);
    logLikelihood = sum(y .* lSigmoid + (1 - y) .* log(1 - exp(lSigmoid)));
    err = - logLikelihood / n;
end