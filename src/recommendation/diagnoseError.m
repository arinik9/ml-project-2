function errors = diagnoseError(y, yHat)
% DIAGNOSEERROR Look at the error made VS the quantity of available data
% INPUT:
%   y: Real results (each data example can have several outputs)
%   yHat: Predicted result
% OUTPUT:
%   errors: [number of available listening count, error made] (sorted)
  [~, aIdx] = find(y);

  % Error made for each artist
  artistsIdx = unique(aIdx);
  d = length(artistsIdx);
  errors = zeros(d, 2);
  for i = 1:d
    errors(i, 1) = nnz(y(:, artistsIdx(i)));
    errors(i, 2) = computeRmse(y(:, artistsIdx(i)), yHat(:, artistsIdx(i)));
  end;

  % Sort by number of ratings available
  errors = sortrows(errors, 1);
  
  % Take average for each number of observed data points
  nCounts = unique(errors(:, 1));
  n = length(nCounts);
  averaged = zeros(n, 2);
  for i = 1:n
      idx = (errors(:, 1) == nCounts(i));
      errs = errors(idx, 2);
      averaged(i, :) = [mean(errs), std(errs)];
  end;
  
  %errorbar(1:n, averaged(:, 1), averaged(:, 2));
  semilogx(nCounts, averaged(:, 1), '.');
  set(gca,'Xdir','reverse')
  title('Average error made over artists with a given number of observations');
  xlabel('Number of available listening counts');
  ylabel('Average RMSE error');
  
  % Spot the largest errors
  largeRmse = 2;
  largeErrorIdx = (errors(:, 2) > largeRmse);
  maxAvailable = max(errors(largeErrorIdx, 1));
  medianAvailable = median(errors(largeErrorIdx, 1));
  fprintf('RMSE > %f occurs over artists having\nat most %d ratings available (median: %d rating available).\n', largeRmse, maxAvailable, medianAvailable);
end
