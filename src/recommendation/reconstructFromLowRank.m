function Yhat = reconstructFromLowRank(Y, U, M)
% RECONSTRUCTFROMLOWRANK Approximate nonzero entries of Y from low-rank approximation
%
% INPUT:
%   Y: Sparse matrix which nonzero entries we want to approximate
%   U, M: Low-rank approximation from ALS-WR
% OUTPUT:
%   Yhat: Reconstructed sparse matrix (estimating the original Y)
    
    % We compute only nonzero entries of Y
    [rIdx, cIdx] = find(Y);
    values = zeros(length(rIdx), 1);
    for i = 1:length(rIdx)
        values(i) = U(:, rIdx(i))' * M(:, cIdx(i));
    end;

    [N, D] = size(Y);
    Yhat = sparse(rIdx, cIdx, values, N, D);
end