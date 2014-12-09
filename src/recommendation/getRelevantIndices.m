function [indices, sizes] = getRelevantIndices(Ytest, Ytrain)
% GETRELEVANTINDICES Find indices of nonzero entries
%
% INPUT
%   Ytest
%   Ytrain
% OUTPUT
%   indices: a cell array holding the unique nonzero users'
%            and artists' indices, as well as the result of find()
%            for both the test and train sets
%   sizes: dimensions of the test and train sets

    indices = {};
    indices.tr = {}; indices.te = {};
    indices.tr.unique = {}; indices.te.unique = {};

    % Indices of nonzero elements
    [indices.tr.u, indices.tr.a] = find(Ytrain);
    [indices.te.u, indices.te.a] = find(Ytest);
    indices.tr.unique.u = unique(indices.tr.u);
    indices.tr.unique.a = unique(indices.tr.a);
    indices.te.unique.u = unique(indices.te.u);
    indices.te.unique.a = unique(indices.te.a);

    % Dimensions
    sizes = {};
    sizes.tr = {}; sizes.te = {};
    sizes.tr.unique = {}; sizes.te.unique = {};

    sizes.tr.nnz = nnz(Ytrain);
    sizes.te.nnz = nnz(Ytest);
    [sizes.tr.u, sizes.tr.a] = size(Ytrain);
    [sizes.te.u, sizes.te.a] = size(Ytest);
    sizes.tr.unique.u = length(indices.tr.unique.u);
    sizes.tr.unique.a = length(indices.tr.unique.a);
    sizes.te.unique.u = length(indices.te.unique.u);
    sizes.te.unique.a = length(indices.te.unique.a);
end
