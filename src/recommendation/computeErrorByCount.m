function [errorByCount, errors] = computeErrorByCount(y, yHat)
%
%
% OUTPUT
%   errorByCount: average and deviation RMSE for each count
%   errors: acual RMSE
  % TODO: allow to choose dimension along which to compute error

  % Error made for each artist
  d = size(y, 2);
  errors = zeros(d, 2);
  for i = 1:d
    errors(i, 1) = nnz(y(:, i));
    errors(i, 2) = computeRmse(y(:, i), yHat(:, i));
  end;
  
  % Sort by number of ratings available
  errors = sortrows(errors, 1);
  
  % Take average for each number of observed data points
  nCounts = unique(errors(:, 1));
  n = length(nCounts);
  errorByCount = zeros(n, 2);
  for i = 1:n
      idx = (errors(:, 1) == nCounts(i));
      errs = errors(idx, 2);
      errorByCount(i, :) = [mean(errs), std(errs)];
  end;
  
end
