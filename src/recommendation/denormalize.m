function [Y] = denormalize(Ynormalized, idx)
% DENORMALIZE Inverse the log transform which was applied.
%
% See `normalizedByUser`.
%
% INPUTS
%  Ynormalized  The (sparse) normalized dataset
%  idx          Initially nonzero indices
% OUTPUTS
%  Y            The matrix, rescaled and recentered

    [N, D] = size(Ynormalized);

    % Note that some values initially non-null in the original Y
    % may have been nulled out in the process of normalization
    values = zeros(size(idx.a, 1), 1);
    for i = 1:length(idx.unique.u)
        user = idx.unique.u(i);
        ii = (idx.u == user);
        values(ii) = exp(Ynormalized(user, idx.a(ii)));
    end;


    Y = sparse(idx.u, idx.a, values, N, D);
end
