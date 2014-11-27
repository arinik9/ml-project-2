function [correlatedVariables, correlations] = findCorrelations(selector, X, y)
% selector: function(x) => boolean selecting the features with respect to
% their correlation, for example:
%     threshold = 0.5;
%     selector = @(x) abs(x) > threshold;
%
% OUTPUT:
%   correlatedVariables = [
%       first feature's index,
%       second feature's index,
%       correlation between the two
%   ]
%   With a line for each couple of variables which correlation matches the
%   given `selector`.
%
%   correlations = raw result of corr(X, y) or corr(X, X) if y is omitted.
    
    % If y is passed as parameter we are looking for correlations between
    % input and output variables.
    % Otherwise we are looking for correlations between features.
    if (nargin < 3)
        correlations = corr(X);
    else
        correlations = corr(X,y);
    end
     

    [corrI, corrJ] = find(selector(correlations));
    idx = (corrI - corrJ > 0);
    
    correlatedVariables = [corrI(idx) corrJ(idx)];
    
    % No features matched the selector
    if(length(correlatedVariables) <= 0)
        return;
    end;
    
    for i = 1:length(correlatedVariables)
        correlatedVariables(i, 3) = correlations(correlatedVariables(i, 1), correlatedVariables(i, 2));
    end;
    
    correlatedVariables = sortrows(correlatedVariables, [-3, 1, 2]);
end


