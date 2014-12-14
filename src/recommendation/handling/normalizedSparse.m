function [Ynormalized, overallMean, overallStd] = normalizedSparse(Y, overallMean, overallStd)
% NORMALIZEDSPARSE Gaussianize the data by applying the log transform
% The data is also centered (global mean only, not per user or per artist).

    [idx, sz] = getRelevantIndices(Y);

    logCounts = log(nonzeros(Y));

    if(~exist('overallMean', 'var'))
        overallMean = mean(logCounts);
    end;
    if(~exist('overallStd', 'var'))
        overallStd = std(logCounts);
    end;

    % We add a tiny noise offset
    % To avoid log(1) yielding new 0 values (which would then be
    % interpreted as unknown).
    normalizedValues = (logCounts - overallMean + 1e-8) ./ 1; %overallStd;

    Ynormalized = sparse(idx.u, idx.a, normalizedValues, sz.u, sz.a);
end
