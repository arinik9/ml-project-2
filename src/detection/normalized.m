function [y, y_test] = normalized(x, x_test)
  % need to keep means and deviations to use them for test data
  % normalization
  means = mean(x);
  deviations = std(x);
  onesX = ones(size(x,1), 1);
  y = (x - onesX * means) ./ (onesX * deviations);
  
  % normalization of the test data with the previous means and deviations
  if (nargin > 1)
      onesX = ones(size(x_test,1), 1);
      y_test = (x_test - onesX * means) ./ (onesX * deviations);
  end
end
