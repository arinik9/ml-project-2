function [Yclean] = removeOutliers(Y, nDev)
% REMOVEOUTLIERS Simple outliers removal for the recommendation dataset
% Warning: since the number of elements change, update any list of indices
% you might have computed before.
%
% INPUTS
%  Y       The initial dataset (sparse matrix)
%  nDev    The maximum number of standard deviations away from the median
%
% OUTPUTS
%  Yclean  The sparse matrix with suspected outliers triplets set to 0
    allCounts = nonzeros(Y);

    % We consider a count is an outlier if it deviates from the global median by
    % more than `nDev` times the global standard deviation.
    % Note that we can only deviate in the positive direction).
    globalMedian = median(allCounts);
    globalDeviation = std(allCounts);
    outliers = (Y > globalMedian + nDev * globalDeviation);
    
    Yclean = Y;
    Yclean(outliers) = 0;
end