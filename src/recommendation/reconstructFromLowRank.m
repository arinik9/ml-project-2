function Yhat = reconstructFromLowRank(U, M, idx, sz)
% RECONSTRUCTFROMLOWRANK Approximate nonzero entries of Y from low-rank approximation
%
% INPUT:
%   U, M: Low-rank approximation from ALS-WR
%   idx: Relevant indices to reconstruct
%   sz: Size of the output target matrix
% OUTPUT:
%   Yhat: Reconstructed sparse matrix (estimating the original Y)
    
    % We compute only nonzero entries of Y
    values = zeros(sz.nnz, 1);
    % TODO: could most likely be optimized
    for i = 1:sz.nnz
        values(i) = U(:, idx.u(i))' * M(:, idx.a(i));
    end;

    Yhat = sparse(idx.u, idx.a, values, sz.u, sz.a);
end