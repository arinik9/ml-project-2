function [Ytest_strong, Ytest_weak, Gstrong, Ytrain, Gtrain] = ...
    getTrainTestSplit(Yoriginal, Goriginal, weakRatio, strongRatio, nDev)

    if(~exist('nDev', 'var'))
        nDev = 3;
    end;

    [Ytest_strong, Ytest_weak, Gstrong, Ytrain, Gtrain] = ...
        splitData(Yoriginal, Goriginal, strongRatio, weakRatio);


    % Normalization & outliers removal
    [Ytrain, overallMean] = normalizedSparse(Ytrain);
    Ytest_weak = normalizedSparse(Ytest_weak, overallMean);
    Ytest_strong = normalizedSparse(Ytest_strong, overallMean);

    Ytrain = removeOutliersSparse(Ytrain, nDev);
end
