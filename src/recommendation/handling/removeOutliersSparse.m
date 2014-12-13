function [Yclean, Y2clean] = removeOutliersSparse(Y, nDev, Y2)
% REMOVEOUTLIERS Simple outliers removal for the recommendation dataset
% Warning: since the number of elements change, update any list of indices
% you might have computed before.
%
% INPUTS
%  Y       The initial dataset (sparse matrix)
%  nDev    The maximum number of standard deviations away from the median
%  [Y2]    A test dataset to treat with the same threshold as Y
%
% OUTPUTS
%  Yclean  The sparse matrix with suspected outliers triplets set to 0
%  [Y2clean] The second matrix cleaned using Y's criterion
    allCounts = nonzeros(Y);

    % We consider a count is an outlier if it deviates from the global median by
    % more than `nDev` times the global standard deviation.
    % Note that we can only deviate in the positive direction).
    globalMedian = median(allCounts);
    globalDeviation = std(allCounts);
    outliers = (Y > globalMedian + nDev * globalDeviation);
    
    Yclean = Y;
    Yclean(outliers) = 0;
    
    if(exist('Y2', 'var'))
        outliers = (Y2 > globalMedian + nDev * globalDeviation);
    
        Y2clean = Y2;
        Y2clean(outliers) = 0;
    end;
end