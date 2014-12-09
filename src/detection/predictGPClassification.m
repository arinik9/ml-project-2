function gpPred = predictGPClassification(gpModel, Xte)
% Takes a trained model of Gaussian Process large scale classification and
% predict output labels for a given test dataset.
% Note on @gp function outputs: The last output argument is a vector of 
% log probabilities lp. The fist four output arguments of the function are mean 
% and variance for the targets and corresponding latent variables respectively.

    n = size(Xte,1);
    
    [~, ~, ~, ~, lp] = gp(gpModel.hyp, gpModel.inffunc, gpModel.meanfunc, gpModel.covfuncF, gpModel.likfunc, gpModel.Xtr, gpModel.ytr, Xte, ones(n,1));

    % lp gives the logarithm probabilities. Get the exp to have predictions
    gpPred = exp(lp);

end