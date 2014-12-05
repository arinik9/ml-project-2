function gpPred = GPClassificationPrediction(y, Xtr, Xte)
% Takes input and output training data, test data and gives prediction with
% Gaussian Process large scale classification
% TODO: Hyper-parameters and model selection in function arguments

    % Mean function
    meanfunc = @meanConst; 
    hyp.mean = 0;

    % use inducing points u and to base the computations on cross-covariances 
    % between training, test and inducing points only
    nu = fix(n); iu = randperm(n); iu = iu(1:nu); u = x(iu,:);

    % Covariance function
    covfunc = @covSEiso; 
    ell = 1.0; % characteristic length-scale
    sf = 1.0; % standard deviation of the signal sf
    hyp.cov = log([ell sf]);
    covfuncF = {@covFITC, {covfunc}, u};

    % Likelihood function
    likfunc = @likErf;

    inffunc = @infFITC_EP;                       % also @infFITC_Laplace is possible

    hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfuncF, likfunc, Xtr, y);
    [a, b, c, d, lp] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, Xtr, y, Xte, ones(n,1));

    % Note: Output arguments: 
    % When computing test probabilities, we call gp with additional test inputs, 
    % and as the last argument a vector of targets for which the log probabilities 
    % lp should be computed. The fist four output arguments of the function are mean 
    % and variance for the targets and corresponding latent variables respectively.

    % lp gives the logarithm probabilities. Get the exp to have predictions
    gpPred = exp(lp);

end