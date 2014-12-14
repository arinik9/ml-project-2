function [Ynormalized, newMean] = normalizedSparse(Y)
% NORMALIZEDSPARSE Gaussianize the data by applying the log transform
    % Add a tiny noise offset
    % To avoid log(1) yielding new 0 values (which would then be
    % interpreted as unknown).
    logCounts = log(nonzeros(Y)) + 1e-8;
    newMean = mean(logCounts);

    [N, D] = size(Y);
    [uIdx, aIdx] = find(Y);
    Ynormalized = sparse(uIdx, aIdx, logCounts, N, D);
end
