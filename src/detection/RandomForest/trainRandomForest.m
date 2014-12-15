function rfModel = trainRandomForest(y, X, nTrees, nVarSample, nMinLeaf)
% Wrapper to be consistent with other models
    
    if (nargin < 5)
       nMinLeaf = 1; % Default value for classification 
    end
    if (nargin < 4)
       nVarSample = sqrt(size(X,2)); % Default value for classification 
    end

    rfModel = TreeBagger(nTrees, X, y, 'MinLeaf', nMinLeaf, 'NVarToSample', nVarSample);
end