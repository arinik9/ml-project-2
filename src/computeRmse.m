function cost = computeRmse(y, yHat)
% y: Real results (each data example can have several outputs)  
% yHat: Predicted result
    cost = sqrt(2 * computeMse(y, yHat));
end

function err = computeMse(y, yHat)
    [n, d] = size(y);
    % Matrix of residuals (n x d)
    residuals = y - yHat;
    % Mean squared error for each output variable (1 x d)
    % TODO: double check this is the correct way
    e = diag(residuals' * residuals)' / (2 * n);
    % Overall MSE (1 x 1)
    err = sum(e) / d;
end
