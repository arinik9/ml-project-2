function rfModel = trainRandomForest(y, X, nTrees, nMinLeaf)
% Wrapper to be consistent with other models
% TODO: add other hyper parameters
    
    if (nargin < 4)
       nMinLeaf = 1; % Default value for classification 
    end

    rfModel = TreeBagger(nTrees, X, y, 'MinLeaf', nMinLeaf);
end