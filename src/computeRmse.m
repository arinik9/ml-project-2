function cost = computeRmse(y, yHat)
% y: Real results (each data example can have several outputs)  
% yHat: Predicted result
    cost = sqrt(2 * computeMse(y, yHat));
end

function err = computeMse(y, yHat)
    % Vector of residuals (nnz x 1)
    residuals = nonzeros(y - yHat);
    % Overall MSE (1 x 1)
    err = sum(residuals .^ 2) / nnz(y);
end
