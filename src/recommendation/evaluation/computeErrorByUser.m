function errorByUser = computeErrorByUser(y, yHat)
% Error made for each user, ordered by true norm(user)
%
% OUPTUT
%   errorByUser: [true norm, predicted norm] Sorted by true norm

    [idx, sz] = getRelevantIndices(y);
    % [norm, error made]
    errorByUser = zeros(sz.unique.u, 2);
    for i = 1:sz.unique.u
        user = idx.unique.u(i);
        errorByUser(i, 1) = norm(nonzeros(y(user, :)));
        errorByUser(i, 2) = norm(nonzeros(yHat(user, :)));
    end;

    [errorByUser, ~] = sortrows(errorByUser);
end
