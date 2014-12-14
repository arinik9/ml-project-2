loadDataset;

weakRatio = 0.2;
strongRatio = 0;
nDev = 3;

setSeed(1);
[~, Ytest, ~, Ytrain, Gtrain] = ...
            getTrainTestSplit(Yoriginal, Goriginal, weakRatio, strongRatio, nDev);
[idx, sz] = getRelevantIndices(Ytest);
[userDV, artistDV] = generateDerivedVariables(Ytest);
