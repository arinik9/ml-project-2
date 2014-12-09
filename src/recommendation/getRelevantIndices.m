function [indices, sizes] = getRelevantIndices(Ytrain, Ytest)
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
    indices.tr = {}; indices.tr.unique = {};

    sizes = {};
    sizes.tr = {}; sizes.tr.unique = {};

    % ----- Train set
    [indices.tr.u, indices.tr.a] = find(Ytrain);
    indices.tr.unique.u = unique(indices.tr.u);
    indices.tr.unique.a = unique(indices.tr.a);

    sizes.tr.nnz = nnz(Ytrain);
    [sizes.u, sizes.a] = size(Ytrain); % Shorthand
    [sizes.tr.u, sizes.tr.a] = size(Ytrain);
    sizes.tr.unique.u = length(indices.tr.unique.u);
    sizes.tr.unique.a = length(indices.tr.unique.a);

    % ----- Test set
    if(exist('Ytest', 'var'))
        indices.te = {}; indices.te.unique = {};
        [indices.te.u, indices.te.a] = find(Ytest);
        indices.te.unique.u = unique(indices.te.u);
        indices.te.unique.a = unique(indices.te.a);

        sizes.te.nnz = nnz(Ytest);
        sizes.te = {}; sizes.te.unique = {};
        [sizes.te.u, sizes.te.a] = size(Ytest);
        sizes.te.unique.u = length(indices.te.unique.u);
        sizes.te.unique.a = length(indices.te.unique.a);
    end;
end
