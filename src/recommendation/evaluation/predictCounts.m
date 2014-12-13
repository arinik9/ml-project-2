function Yhat = predictCounts(predict, idx, sz)
% PREDICTCOUNTS Generate a sparse matrix of predictions
%
% INPUT
%   predict: function(user index, artist index)
%            which returns the corresponding prediction
%            in log space.
%   idx:     indices to predict on
%   sz:      expected size of the output matrix
% OUTPUT
%   Yhat:    sparse matrix of predictions

    % For each required (user, artist) pair
    values = zeros(length(idx.u), 1);
    parfor i = 1:length(idx.u)
        user = idx.u(i);
        artist = idx.a(i);
        values(i) = predict(user, artist);
    end;

    Yhat = sparse(idx.u, idx.a, values, sz.u, sz.a);
end
