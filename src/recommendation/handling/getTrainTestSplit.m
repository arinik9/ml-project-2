function [Ytest_strong, Ytest_weak, Gstrong, Ytrain, Gtrain] = ...
    getTrainTestSplit(Yoriginal, Goriginal, weakRatio, strongRatio, nDev)

    if(~exist('nDev', 'var'))
        nDev = 3;
    end;

    [Ytest_strong, Ytest_weak, Gstrong, Ytrain, Gtrain] = ...
        splitData(Yoriginal, Goriginal, strongRatio, weakRatio);


    % Normalization & outliers removal
    Ytrain = normalizedSparse(Ytrain);
    Ytest_weak = normalizedSparse(Ytest_weak);
    Ytrain = removeOutliersSparse(Ytrain, nDev);
end
