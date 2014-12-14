% Script used to test different SVM settings
% First we try out different kernel types using CV for validation
% Then RBF being the best suited kernel, we try to find the best gamma
% parameter

getStartedDetection;

%  No need to scale features since all the features are in the range [0, 0.2]

%% Trying out on pca(X) and pca(exp(X))    
    
plot_flag = 0;
predict = @(model, X) predictSVM(model, X);    
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);

learn = @(y, X) trainSVM(y, X, '-t 2 -b 1 -e 0.01');
[svmPCA.trAvgTPR, svmPCA.teAvgTPR, svmPCA.predTr, svmPCA.predTe, svmPCA.trueTr, svmPCA.trueTe] = kFoldCrossValidation(y, pcaX, 3, learn, predict, computePerformance, 'SVM RBF kernel + Exp');

learn = @(y, X) trainSVM(y, X, '-t 2 -b 1 -e 0.01');
[svmPCAexp.trAvgTPR, svmPCAexp.teAvgTPR, svmPCAexp.predTr, svmPCAexp.predTe, svmPCAexp.trueTr, svmPCAexp.trueTe] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, 'SVM RBF kernel + Exp');

% Compare results
methodNames = {'pca(X)', 'pca(exp(X))'};
avgTPRList = kCVevaluateMultipleMethods( cat(3, svmPCA.trueTe, svmPCAexp.trueTe), cat(3, svmPCA.predTe, svmPCAexp.predTe), true, methodNames );

    
%% kCV on different models
plot_flag = 0;
predict = @(model, X) predictSVM(model, X);
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);

% Linear kernel
learn = @(y, X) trainSVM(y, X, '-t 0 -b 1 -e 0.01');
[svm1.trAvgTPR, svm1.teAvgTPR, svm1.predTr, svm1.predTe, svm1.trueTr, svm1.trueTe] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, 'SVM Linear kernel');
 
% Quadratic kernel
learn = @(y, X) trainSVM(y, X, '-t 1 -b 1 -e 0.01');
[svm2.trAvgTPR, svm2.teAvgTPR, svm2.predTr, svm2.predTe, svm2.trueTr, svm2.trueTe] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, 'SVM Polynomial kernel');

% RBF kernel
learn = @(y, X) trainSVM(y, X, '-t 2 -b 1 -e 0.01');
[svm3.trAvgTPR, svm3.teAvgTPR, svm3.predTr, svm3.predTe, svm3.trueTr, svm3.trueTe] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, 'SVM RBF kernel');

%% Compare SVM with different kernels

methodNames = {'Linear kernel', 'Polynomial kernel', 'RBF kernel'};
avgTPRList = kCVevaluateMultipleMethods( cat(3, svm1.trueTe, svm2.trueTe, svm3.trueTe), cat(3, svm1.predTe, svm2.predTe, svm3.predTe), true, methodNames );

%% Learning: SVM RBF Kernel Gamma parameter
    
gammaValues = [(5/(size(pcaExpX,2))),(2/(size(pcaExpX,2))), (1/size(pcaExpX,2)), (1/(2*size(pcaExpX,2))), (1/(5*size(pcaExpX,2))), (1/(10*size(pcaExpX,2)))];
[bestGamma, testTPR, trainTPR] = findGammaSVM(y, pcaExpX, 3, gammaValues);
savePlot('./report/figures/detection/svm-gamma-learningcurve.pdf','Gamma values','TPR on train (blue) and test (red)');
