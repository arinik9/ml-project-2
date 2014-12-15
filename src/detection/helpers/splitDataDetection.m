function [XTr, yTr, XTe, yTe] = splitDataDetection(y, X, prop, seed)
% randomly split the data into train and test given a proportion
    if(nargin < 4)
        seed = 1;
    end;
    setSeed(seed);
    
    N = size(y,1);
    
    % generate random indices
    idx = randperm(N);
    Ntr = floor(prop * N);
    
    % select few as training and others as testing
    idxTr = idx(1:Ntr);
    idxTe = idx(Ntr+1:end);
    
    % create train-test split
    XTr = X(idxTr, :);
    yTr = y(idxTr);
    XTe = X(idxTe, :);
    yTe = y(idxTe);
end