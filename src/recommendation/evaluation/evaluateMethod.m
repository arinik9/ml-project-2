function [trErrorAvg, teErrorAvg, trError, teError] = evaluateMethod(name, getPredictor, Y, G, nSplits, showErrorDiagnostic)
% EVALUATEMETHOD Evaluate method on many random train / test splits
%
% INPUT
%   name:         Name of the method being evaluated
%   getPredictor: function(Y, Ytest)
%                 which returns a `predict(user, artist)` function
%   nSplits:      number of random train / test splits to use
%                 in order to estimate expected the test error.
%   showErrorDiagnostic: boolean flag
% OUTPUT
%   trError, teError: estimated train and test errors

    if(~exist('showErrorDiagnostic', 'var'))
        showErrorDiagnostic = 0;
    end;
    if(~exist('nSplits', 'var'))
        nSplits = 5;
    end;

    % We want unpredictable random numbers
    setSeed(-1);

    % TODO: vary test / train proportions
    weakRatio = 0.1;
    % TODO: handle strong prediction here as well?
    strongRatio = 0;
    % TODO: test removing more or less "outliers"
    nDev = 3;

    e = {};
    e.tr = zeros(nSplits, 1);
    e.te = zeros(nSplits, 1);

    for i = 1:nSplits
        % Generate train / test splits
        [~, Ytest, ~, Ytrain, Gtrain] = ...
            getTrainTestSplit(Y, G, weakRatio, strongRatio, nDev);

        if(showErrorDiagnostic && i == 1)
            [e.tr(i), e.te(i), trYhat, teYhat] = ...
                evaluateMethodOnce(getPredictor, Ytrain, Ytest, Gtrain);
            
            % TODO: adapt diagnostic to support several runs
            diagnoseError(Ytrain, trYhat, Ytest, teYhat);
        else
            [e.tr(i), e.te(i), ~, ~] = evaluateMethodOnce(getPredictor, Ytrain, Ytest, Gtrain);
        end;
        
        fprintf('%s [split %d]: %f | %f\n', name, i, e.tr(i), e.te(i));
    end;

    % Get back all train and test error to plot error
    trError = e.tr;
    teError = e.te;
    
    trErrorAvg = mean(e.tr);
    teErrorAvg = mean(e.te);
    
    fprintf('----- %s [average]: %f | %f\n\n', name, trErrorAvg, teErrorAvg);
end
