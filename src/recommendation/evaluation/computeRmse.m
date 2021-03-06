function cost = computeRmse(y, yHat)
% y: Real results (each data example can have several outputs)
% yHat: Predicted result
    cost = sqrt(2 * computeMse(y, yHat));
end

function err = computeMse(y, yHat)
% COMPUTEMSE Compute error only on nonzero entries of y
    % Vector of residuals (nnz x 1)
    idx = (y ~= 0);
    residuals = nonzeros(y(idx) - yHat(idx));
    % Overall MSE (1 x 1)
    err = sum(residuals .^ 2) / (2 * nnz(y));
end
