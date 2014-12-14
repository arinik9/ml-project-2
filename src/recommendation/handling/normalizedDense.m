function Ynormalized = normalizedDense(Y)
% NORMALIZEDDENSE Normalize for each feature (column)
    [n, ~] = size(Y);

    means = mean(Y, 1);
    deviations = std(Y, 1);
    Ynormalized = (Y - repmat(means, [n 1])) ./ repmat(deviations, [n 1]);
end
