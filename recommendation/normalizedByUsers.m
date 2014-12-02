function [Ynormalized, meanPerUser, deviationPerUser] = normalizedByUsers(Y)
% NORMALIZEDBYUSERS Normalization among users for the recommendation dataset
%
% INPUTS
%  Y            The initial dataset (sparse matrix)
% OUTPUTS
%  Ynormalized  A sparse normalized version of the input
%  meanPerUser  Means used to normalize the rows (keep it in order to
%               denormalize after producing predictions) 
%  deviationPerUser Deviation measure used to normalize the rows (keep it
%                   in order to denormalize after producing predictions) 

    [uIdx, aIdx] = find(Y);
    % Indices of users having at least one data example
    usersIdx = unique(uIdx);

    % We normalize *among user*, that is we make users lie on the same scale
    % (a heavy user will have counts comparable to a light user).
    % This way, we retain the artists popularity, which is important information.
    meanPerUser = zeros(size(usersIdx, 1), 1);
    deviationPerUser = zeros(size(usersIdx, 1), 1);

    values = zeros(nnz(Y), 1);
    zeroIdx = [];
    for i = 1:length(usersIdx)
        counts = nonzeros(Y(i, :));
        meanPerUser(i) = mean(counts);
        deviationPerUser(i) = std(counts);
        if(length(counts) <= 1 || deviationPerUser(i) == 0)
            zeroIdx = [zeroIdx; i];
        end;
    end;
    
    % If we have only 1 data point for a user, or only equal values std() is 0
    % (and we don't want to divide by 0).
    % We replace by the deviation to the mean over all counts, as an estimate of
    % the actual deviation we might have observed).
    % Even if it's a bit hacky, the impact is small (we have very few such cases)
    o = ones(nnz(zeroIdx), 1);
    deviationPerUser(zeroIdx) = std([meanPerUser(zeroIdx)'; mean(meanPerUser) * o']);
    meanPerUser(zeroIdx) = mean(meanPerUser) * o;
    % Generate a new normalized sparse matrix
    for i = 1:length(usersIdx)
        values(uIdx == usersIdx(i)) = (nonzeros(Y(i, :)) - meanPerUser(i)) ./ deviationPerUser(i);
    end;
    Ynormalized = sparse(uIdx, aIdx, values, size(Y, 1), size(Y, 2));

end