function [Ytest_strong, Ytest_weak, Gstrong, Ytrain, Gtrain] = ...
    splitData(Y, G, testRatioStrong, testRatioWeak)
% SPLITDATA Generate a random test/train split for both weak and strong prediction
% INPUT:
%   Y:               original listening counts matrix
%   G:               original social network
%   testRatioStrong: proportion of users to keep completely hidden
%                    to use them for strong prediction testing
%   testRatioWeak:   proportion of available counts to withhold for each user
%                    to use them for weak prediction testing
% OUTPUT:
%   Ytest_strong: pairs (user, artist) = count for new users to test
%   Ytest_weak: pairs (user, artist) = hidden counts for existing users to test
%   Gstrong: friendship graph of new users with all users
%   Ytrain and Gtrain is the new training set
    if(nargin < 3)
        testRatioStrong = 0.1;
    end;
    if(nargin < 4)
        testRatioWeak = 0.1;
    end;

    % ----- Test data for Strong generalization
    % Keep testRatioStrong percent of users hidden for testing as 'new users'

    % Total number of available users
    nU = size(Y,1);
    idx = randperm(nU);
    % Number of users to withhold
    nTe = floor(nU * testRatioStrong);
    % Corresponding indices
    idxTe = idx(1:nTe);
    idxTr = idx(nTe+1:end);
    % Split the matrices (counts and social network)
    Ytrain = Y(idxTr,:);
    Ytest_strong = Y(idxTe,:);
    Gtrain = G(idxTr, idxTr);
    % WARNING, the whole right part of Gstrong is useless to us:
    % is represents friendships between unseen users,
    % we don't have any data about them.
    Gstrong = G(idxTe, [idxTr idxTe]);

    % ----- Test data for weak generalization
    % Keep testRatioWeak percent of entries per remaining user as test data
    [nU, nA] = size(Ytrain);

    userTestIndices = [];
    artistTestIndices = [];
    countTestValues = [];
    % For each user
    for i = 1:nU
        % Find available counts for this user
        available = find(Ytrain(i, :) ~= 0)';
        % Number of counts to withhold for this user
        numA = floor(length(available) * testRatioWeak);
        % Make sure to leave some counts for prediction
        if (length(available) - numA > 0) && (numA > 0)
            % Pick some of those counts for testing
            ind = randperm(length(available));
            j = available(ind(1:numA));

            userTestIndices = [userTestIndices; i * ones(numA, 1)];
            artistTestIndices = [artistTestIndices; j];
            countTestValues = [countTestValues; Ytrain(i, j)'];

            %fprintf('User %i has %d counts, we hide %d.\n', i, length(available), numA);
        end
    end
    Ytest_weak = sparse(userTestIndices, artistTestIndices, countTestValues, nU, nA);
    % Hide the extracted counts in the remaining test set
    Ytrain(sub2ind([nU nA], userTestIndices, artistTestIndices)) = 0;

    %fprintf('We hide %d counts in total for weak prediction.\n', nnz(countTestValues));
end
