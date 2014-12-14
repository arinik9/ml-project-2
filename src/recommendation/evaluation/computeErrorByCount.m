function [errorByCount, errors] = computeErrorByCount(y, yHat, yRef)
%
%
% OUTPUT
%   errorByCount: average and deviation RMSE for each count
%   errors: acual RMSE
  % TODO: allow to choose dimension along which to compute error

    if(~exist('yRef', 'var'))
        yRef = y;
    end;

    % Error made for each artist
    [idx, sz] = getRelevantIndices(y);
    errors = zeros(sz.unique.a, 2);
    for j = 1:sz.unique.a
        artist = idx.unique.a(j);
        errors(j, 1) = nnz(yRef(:, artist));
        errors(j, 2) = computeRmse(y(:, artist), yHat(:, artist));
    end;

    % Sort by number of ratings available
    errors = sortrows(errors, 1);

    % Take average for each number of observed data points
    nCounts = unique(errors(:, 1));
    n = length(nCounts);
    errorByCount = zeros(n, 2);
    for j = 1:n
      idx = (errors(:, 1) == nCounts(j));
      errs = errors(idx, 2);
      errorByCount(j, :) = [mean(errs), std(errs)];
    end;

end
