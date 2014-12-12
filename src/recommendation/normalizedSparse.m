function [Ynormalized, newMean] = normalizedSparse(Y)
% NORMALIZEDSPARSE Gaussianize the data by applying the log transform

    logCounts = log(nonzeros(Y));
    newMean = mean(logCounts);

    [N, D] = size(Y);
    [uIdx, aIdx] = find(Y);
    Ynormalized = sparse(uIdx, aIdx, logCounts, N, D);
end
