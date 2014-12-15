loadDataset;
% Number of random train / test splits to generate
% TODO: moar
nSplits = 2;

% Shortcut
evaluate = @(name, learn) evaluateMethod(name, learn, Yoriginal, Goriginal, nSplits, 1);

%% Lambda values on ALSWR

nFeatures = 50; % Target reduced dimensionality
displayLearningCurve = 1;
name = 'ALSWR';

lambda1 = 0.01;
learnAlsWr1 = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda1, displayLearningCurve);
[e.tr.(name), e.te.(name), trLambda1, teLambda1] = evaluate(name, learnAlsWr1);

lambda2 = 0.025;
learnAlsWr2 = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda2, displayLearningCurve);
[e.tr.(name), e.te.(name), trLambda2, teLambda2] = evaluate(name, learnAlsWr2);

lambda3 = 0.05;
learnAlsWr3 = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda3, displayLearningCurve);
[e.tr.(name), e.te.(name), trLambda3, teLambda3] = evaluate(name, learnAlsWr3);

lambda4 = 0.1;
learnAlsWr4 = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda4, displayLearningCurve);
[e.tr.(name), e.te.(name), trLambda4, teLambda4] = evaluate(name, learnAlsWr4);

lambda5 = 0.2;
learnAlsWr5 = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda5, displayLearningCurve);
[e.tr.(name), e.te.(name), trLambda5, teLambda5] = evaluate(name, learnAlsWr5);

lambda6 = 0.5;
learnAlsWr6 = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda6, displayLearningCurve);
[e.tr.(name), e.te.(name), trLambda6, teLambda6] = evaluate(name, learnAlsWr5);


boxplot([teLambda1 teLambda2 teLambda3 teLambda4 teLambda5 teLambda6], 'labels', {'0.01','0.02','0.05','0.1','0.2','0.5'});
savePlot('./report/figures/recommendation/alswr-lambda-learningcurve-test.pdf','Lambda','RMSE');
boxplot([trLamda1 trLambda2 trLambda3 trLambda4 trLamda5 trLambda6], 'labels', {'0.01','0.02','0.05','0.1','0.2','0.5'});
savePlot('./report/figures/recommendation/alswr-lambda-learningcurve-train.pdf','Lambda','RMSE');


%% # Features on ALSWR

lambda = 0.5;
displayLearningCurve = 0;

nFeatures = 50; % Target reduced dimensionality
learnAlsWr = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda, displayLearningCurve);
name = 'ALSWR';
[e.tr.(name), e.te.(name), trFeatures1, teFeatures1] = evaluate(name, learnAlsWr);

nFeatures = 100;
learnAlsWr = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda, displayLearningCurve);
[e.tr.(name), e.te.(name), trFeatures2, teFeatures2] = evaluate(name, learnAlsWr);

nFeatures = 250;
learnAlsWr = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda, displayLearningCurve);
[e.tr.(name), e.te.(name), trFeatures3, teFeatures3] = evaluate(name, learnAlsWr);

nFeatures = 500;
learnAlsWr = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda, displayLearningCurve);
[e.tr.(name), e.te.(name), trFeatures4, teFeatures4] = evaluate(name, learnAlsWr);

nFeatures = 30; % Target reduced dimensionality
learnAlsWr = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda, displayLearningCurve);
name = 'ALSWR';
[e.tr.(name), e.te.(name), trFeatures5, teFeatures5] = evaluate(name, learnAlsWr);

nFeatures = 1000;
learnAlsWr = @(Y, Ytest, userDV, artistDV) learnAlsWrPredictor(Y, Ytest, userDV, artistDV, nFeatures, lambda, displayLearningCurve);
[e.tr.(name), e.te.(name), trFeatures6, teFeatures6] = evaluate(name, learnAlsWr);

boxplot([teFeatures5 teFeatures1 teFeatures2 teFeatures3 teFeatures4 teFeatures6], 'labels', {'30','50','100','250','500','1000'});
savePlot('./report/figures/recommendation/alswr-nfeatures-learningcurve-test.pdf','Nb Features','RMSE');
boxplot([trFeatures5 trFeatures1 trFeatures2 trFeatures3 trFeatures4 trFeatures6], 'labels', {'30','50','100','250','500','1000'});
savePlot('./report/figures/recommendation/alswr-nfeatures-learningcurve-train.pdf','Nb Features','RMSE');

%% K-Means

nFeatures = 20;
lambda = 0.000001;

K = 5;
reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 0)';
getSimilarity = @(Ytrain, Ytest, userDV) computeSimilarityMatrix(Ytrain, Ytest, userDV, reduceSpace);
learnKMeansALS = @(Y, Ytest, userDV, artistDV) ...
    learnKMeansPredictor(Y, Ytest, userDV, artistDV, K, getSimilarity(Y, Ytest, userDV));
name = ['K', int2str(K), 'MeansALS'];
[e.tr.(name), e.te.(name), trK1, teK1] = evaluate(name, learnKMeansALS);

K = 10;
reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 0)';
getSimilarity = @(Ytrain, Ytest, userDV) computeSimilarityMatrix(Ytrain, Ytest, userDV, reduceSpace);
learnKMeansALS = @(Y, Ytest, userDV, artistDV) ...
    learnKMeansPredictor(Y, Ytest, userDV, artistDV, K, getSimilarity(Y, Ytest, userDV));
name = ['K', int2str(K), 'MeansALS'];
[e.tr.(name), e.te.(name), trK2, teK2] = evaluate(name, learnKMeansALS);

K = 15;
reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 0)';
getSimilarity = @(Ytrain, Ytest, userDV) computeSimilarityMatrix(Ytrain, Ytest, userDV, reduceSpace);
learnKMeansALS = @(Y, Ytest, userDV, artistDV) ...
    learnKMeansPredictor(Y, Ytest, userDV, artistDV, K, getSimilarity(Y, Ytest, userDV));
name = ['K', int2str(K), 'MeansALS'];
[e.tr.(name), e.te.(name), trK3, teK3] = evaluate(name, learnKMeansALS);

K = 20;
reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 0)';
getSimilarity = @(Ytrain, Ytest, userDV) computeSimilarityMatrix(Ytrain, Ytest, userDV, reduceSpace);
learnKMeansALS = @(Y, Ytest, userDV, artistDV) ...
    learnKMeansPredictor(Y, Ytest, userDV, artistDV, K, getSimilarity(Y, Ytest, userDV));
name = ['K', int2str(K), 'MeansALS'];
[e.tr.(name), e.te.(name), trK4, teK4] = evaluate(name, learnKMeansALS);

K = 3;
reduceSpace = @(Ytrain, Ytest) alswr(Ytrain, Ytest, nFeatures, lambda, 0)';
getSimilarity = @(Ytrain, Ytest, userDV) computeSimilarityMatrix(Ytrain, Ytest, userDV, reduceSpace);
learnKMeansALS = @(Y, Ytest, userDV, artistDV) ...
    learnKMeansPredictor(Y, Ytest, userDV, artistDV, K, getSimilarity(Y, Ytest, userDV));
name = ['K', int2str(K), 'MeansALS'];
[e.tr.(name), e.te.(name), trK5, teK5] = evaluate(name, learnKMeansALS);


boxplot([teK5 teK1 teK2 teK3 teK4], 'labels', {'3', '5','10','15','20'})
savePlot('./report/figures/recommendation/kmeans-k-learningcurve-test.pdf','K','RMSE');
boxplot([trK5 trK1 trK2 trK3 trK4], 'labels', {'3', '5','10','15','20'})
savePlot('./report/figures/recommendation/kmeans-k-learningcurve-train.pdf','K','RMSE');

%% Head/Tail predictor

headThreshold = 5;
name = ['HeadTail', int2str(headThreshold)];
learnHeadTail = @(Y, Ytest, userDV, artistDV) learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold);

[e.tr.(name), e.te.(name), trErrH1, teErrH1] = evaluate(name, learnHeadTail);

headThreshold = 10;
name = ['HeadTail', int2str(headThreshold)];
learnHeadTail = @(Y, Ytest, userDV, artistDV) learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold);

[e.tr.(name), e.te.(name), trErrH2, teErrH2] = evaluate(name, learnHeadTail);

headThreshold = 50;
name = ['HeadTail', int2str(headThreshold)];
learnHeadTail = @(Y, Ytest, userDV, artistDV) learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold);

[e.tr.(name), e.te.(name), trErrH3, teErrH3] = evaluate(name, learnHeadTail);

headThreshold = 100;
name = ['HeadTail', int2str(headThreshold)];
learnHeadTail = @(Y, Ytest, userDV, artistDV) learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold);

[e.tr.(name), e.te.(name), trErrH4, teErrH4] = evaluate(name, learnHeadTail);

boxplot([teErrH1 teErrH2 teErrH3], 'labels', {'5', '10','50'})
savePlot('./report/figures/recommendation/headtail-threshold-learningcurve-test.pdf','Threshold','RMSE');
boxplot([trErrH1 trErrH2 trErrH3], 'labels', {'5', '10','50'});
savePlot('./report/figures/recommendation/headtail-threshold-learningcurve-train.pdf','Threshold','RMSE');

%% Headtail predictor clusters

headThreshold = 10;
name = ['HeadTail', int2str(headThreshold)];

K = 5;
learnHeadTail = @(Y, Ytest, userDV, artistDV) learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold, K);
[e.tr.(name), e.te.(name), trErrK5, teErrK5] = evaluate(name, learnHeadTail);

K = 10;
learnHeadTail = @(Y, Ytest, userDV, artistDV) learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold, K);
[e.tr.(name), e.te.(name), trErrK10, teErrK10] = evaluate(name, learnHeadTail);

K = 20;
learnHeadTail = @(Y, Ytest, userDV, artistDV) learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold, K);
[e.tr.(name), e.te.(name), trErrK20, teErrK20] = evaluate(name, learnHeadTail);

%%
boxplot([teErrK5 teErrK10 teErrK20], 'labels', {'5', '10','50'})
savePlot('./report/figures/recommendation/headtail-threshold-learningcurve-test.pdf','Threshold','RMSE');
boxplot([trErrK5 trErrK10 trErrK20], 'labels', {'5', '10','50'});
savePlot('./report/figures/recommendation/headtail-threshold-learningcurve-train.pdf','Threshold','RMSE');
