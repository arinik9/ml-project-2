function [Ynormalized, newMean, newStd] = normalizedSparse(Y)
% NORMALIZEDSPARSE Gaussianize the data by applying the log transform
    
    [idx, sz] = getRelevantIndices(Y);
    
    logCounts = log(nonzeros(Y));
    newMean = mean(logCounts);
    newStd = mean(logCounts);
    
    % We add a tiny noise offset
    % To avoid log(1) yielding new 0 values (which would then be
    % interpreted as unknown).
    normalizedValues = (logCounts - newMean + 1e-8) ./ 1; %newStd;
    
    Ynormalized = sparse(idx.u, idx.a, normalizedValues, sz.u, sz.a);
end
