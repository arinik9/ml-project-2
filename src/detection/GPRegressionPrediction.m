function gpPred = GPRegressionPrediction(yTr, Xtr, Xte)

    n = size(Xte, 1);

    % Mean function
    meanfunc = @meanConst; 
    hyp.mean = 0;

    % use inducing points u and to base the computations on cross-covariances 
    % between training, test and inducing points only
    nu = fix(n); iu = randperm(n); iu = iu(1:nu); u = Xtr(iu,:);

    % Covariance function
    covfunc = @covSEiso; 
    ell = 1.0; % characteristic length-scale
    sf = 1.0; % standard deviation of the signal sf
    hyp.cov = log([ell sf]);
    covfuncF = {@covFITC, {covfunc}, u};

    % To try also : 
    % covSEard hyp.cov = log(ones(1, size(u,1) + 1) 1xD+1 size % we should try
    % that one also

    % Likelihood function
    likfunc = @likGauss; 
    sn = 0.1; % standard deviation of the noise
    hyp.lik = log(sn); 

    % compute the (joint) negative log probability (density) nlml (also called marginal likelihood or evidence)
    % nlml = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y);

    hyp = minimize(hyp, @gp, -100, @infFITC, meanfunc, covfuncF, likfunc, Xtr, yTr);
    [m, ~] = gp(hyp, @infFITC, meanfunc, covfuncF, likfunc, Xtr, yTr, Xte);

    % Take a threshold to assign to one of the class
    gpPred = outputLabelsFromPrediction(m, 0);

end