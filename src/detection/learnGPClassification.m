function model = learnGPClassification(y, Xtr, likFunction, infFunction, covFunction, covHyp, nIndPoints)
% Takes input and output training data, test data and build model of
% Gaussian Process large scale classification for binary classification
% problems. Default parameters have been tuned to work on our data.
% Outputs:
%   - model: Gaussian Process Model for large scale classification
% Inputs:
%   - y: train outputs
%   - Xtr: train inputs
%   - Xte: test inputs
%   - likFunction: likelihood function for GP Classification. It can be: 
%       - @likErf: Probit regression (cumulative density function of a
%       standard normal distribution)
%       - @likLogistic: Logistic Function (linear model with logistic
%       response function)
%   - infFunction: Inference Function. As we are doing large scale
%   classification it must be an FITC approximation function:
%       - @infFITC_Laplace: Laplace approximation (which is faster)
%       - @infFITC_EP: Expectation Propagation approximation (which seems
%       more accurate but much slower)
%   - covFunction: covariance function that we are using. So far we have
%   tried:
%       - @covSEiso: Squared Exponential covariance function with isotropic
%       distance measure (less hyperparameters) which acts as the default
%       covariance function
%       - @covSEard: Squared exponential with automatic relevance determination
%       that has one characteristic length-scale parameter for each dimension
%       of the input space, and a signal magnitude parameter. i.e hyp.cov of
%       dimenion 1 x (D+1) (lot of parameters to estimate)
%   - covHyp: hyper parameters associated with the covariance function
%     This may prove useful with our data:
%       - covFunction = @covSEiso; ell = 2.0; sf = 2.0; covHyp = log([ell sf]);
%       - covFunction = @covSEard; ell = 2.0; sf = 2.0; covHyp = log([ell * ones(1,size(Xtr,2)) sf]);
%   - nInductionPoints: number of induction points used. It should follow
%   the order of magnitude of the train input data.
        
    if (nargin < 4)
       % Default likelihood function 
       likFunction = @likLogistic; % or @likErf
    end

    if (nargin < 5)
       % Default inference function: Laplace approximation (faster)
       infFunction = @infFITC_Laplace;
    end
    
    if (nargin < 6)
        % Default covariance function: 
        % Squared Exponential covariance function with 
        % isotropic distance measure (less hyperparameters)
        covFunction = @covSEiso;
    end
    
    if (nargin < 7)
        % Set default hyperparameters for the covariance function
        % Those values work well with our data
        ell = 2.0;
        sf = 2.0;
        covHyp = log([ell sf]); % for @covSEiso
    end
    
    if (nargin < 8)
       % Default number of inducing points
       nIndPoints = floor(size(Xtr,1)/ 50);
    end
    
    % Compute induction points:
    % use inducing points u and to base the computations on cross-covariances 
    % between training, test and inducing points only
    
    % # induction points
    nIpoints = nIndPoints;

    nu = fix(nIpoints);
    fprintf('Large scale classification using %d inducing points\n', nu);
    iu = randperm(nIpoints); iu = iu(1:nu); u = Xtr(iu,:); % was Xte(iu,:);

    % Mean function
    meanfunc = @meanConst; 
    hyp.mean = 0;
    
    % Covariance function
    covfunc = covFunction;
    hyp.cov = covHyp; 

    % Large Scale: wrap the covariance function covfunc into covFITC.m
    covfuncF = {@covFITC, {covfunc}, u}; 

    % Likelihood function
    likfunc = likFunction;
    
    % Inference function
    inffunc = infFunction;
    
    % Minimizing the hyperparameters
    fprintf('Minimize hyperparameters..\n');
    hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfuncF, likfunc, Xtr, y)
    
    % Store the model data to be used for prediction
    model.hyp = hyp;
    model.inffunc = inffunc;
    model.meanfunc = meanfunc;
    model.covfuncF = covfuncF;
    model.likfunc = likfunc;
    model.Xtr = Xtr;
    model.ytr = y;

end