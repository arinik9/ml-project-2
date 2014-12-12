function [indices, sizes] = getRelevantIndices(Y, Ytest)
% GETRELEVANTINDICES Find indices of nonzero entries
%
% INPUT
%   Y
%   [Ytest] Optional (if passed, the train / test distinction will be made)
% OUTPUT
%   indices: a cell array holding the unique nonzero users'
%            and artists' indices, as well as the result of find()
%            for both the test and train sets
%   sizes: dimensions of the test and train sets

    indices = {}; indices.unique = {};
    sizes = {}; sizes.unique = {};

    % ----- One matrix mode (no train / test distinction)
    [indices.u, indices.a] = find(Y);
    indices.unique.u = unique(indices.u);
    indices.unique.a = unique(indices.a);

    sizes.nnz = nnz(Y);
    [sizes.u, sizes.a] = size(Y); % Shorthand
    [sizes.u, sizes.a] = size(Y);
    sizes.unique.u = length(indices.unique.u);
    sizes.unique.a = length(indices.unique.a);

    % ----- Test set
    if (exist('Ytest', 'var') ~= 0)
        indices.tr = indices;
        sizes.tr = sizes;

        indices.te = {}; indices.te.unique = {};
        sizes.te = {}; sizes.te.unique = {};

        [indices.te.u, indices.te.a] = find(Ytest);
        indices.te.unique.u = unique(indices.te.u);
        indices.te.unique.a = unique(indices.te.a);

        sizes.te.nnz = nnz(Ytest);
        [sizes.te.u, sizes.te.a] = size(Ytest);
        sizes.te.unique.u = length(indices.te.unique.u);
        sizes.te.unique.a = length(indices.te.unique.a);
    end;
end
