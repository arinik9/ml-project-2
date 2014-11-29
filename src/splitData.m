function [Ytest_strong, Ytest_weak, Gtest, Ytrain, Gtrain] = ...
    splitData(Y, G, testRatioStrong, testRatioWeak, seed)
% SPLITDATA Generate a random test/train split for both weak and strong prediction
% INPUT:
%   Y:               original listening counts matrix
%   G:               original social network
%   testRatioStrong: proportion of users to keep completely hidden
%                    to use them for strong prediction testing
%   testRatioWeak:   proportion of counts to withhold for each artist
%                    to use them for weak prediction testing
% OUTPUT:
%   Ytest_strong: pairs (user, artist) = count for new users to test
%   Ytest_weak: pairs (user, artist) = count for existing users to test
%   Gstrong: friendship graph of new users with all users
%   Ytrain_new and Gtrain_new is the new training set 
    if(nargin < 3)
        testRatioStrong = 0.1;
    end;
    if(nargin < 4)
        testRatioWeak = 0.1;
    end;
    if(nargin < 5)
        seed = 1;
    end;
    setSeed(seed);

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
    Gtest = G(idxTe, [idxTr idxTe]);

    % ----- Test data for weak generalization
    % Keep testRatioWeak percent of entries per remaining user as test data
    % TODO: double check dimensions
    [nU, nA] = size(Ytrain);
    % Number of counts to withhold per artist
    % TODO: would it be interesting to withhold a proportion of the
    % available counts *for a user* instead? (i.e. the more counts
    % available for a given user, the more we withhold)
    numU = floor(nU * testRatioWeak);
    userTestIndices = [];
    artistTestIndices = [];
    countTestValues = [];
    % For each artist
    for j = 1:nA
        % Find available counts for this artist
        available = find(Ytrain(:, j) ~= 0);
        if length(available) > 2
            % Pick some of those counts for testing
            ind = unidrnd(length(available), numU, 1);
            i = available(ind);
            userTestIndices = [userTestIndices; i];
            artistTestIndices = [artistTestIndices; j * ones(numU,1)];
            countTestValues = [countTestValues; Ytrain(i,j)];
        end
    end
    Ytest_weak = sparse(userTestIndices, artistTestIndices, countTestValues, nU, nA);
    % Hide the extracted counts in the remaining test set
    Ytrain(sub2ind([nU nA], userTestIndices, artistTestIndices)) = 0;
end
