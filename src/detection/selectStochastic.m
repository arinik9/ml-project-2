function [XStochastic, yStochastic] = selectStochastic(y, X, nData, seed)
% randomly split the data into train and test given a proportion
    if(nargin < 4)
        seed = 1;
    end;
    setSeed(seed);
    
    N = size(y,1);
    
    % generate random indices
    idx = randperm(N);
    
    % select subsample of size nData
    idxStochastic = idx(1:nData);
    
    % create subsets
    XStochastic = X(idxStochastic, :);
    yStochastic = y(idxStochastic);
end