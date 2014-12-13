getStartedDetection;

%% SVM Model
%  no need to scale features since all the features are in the range [0, 0.2]


	% options:
	% -s svm_type : set type of SVM (default 0)
	% 	0 -- C-SVC
	% 	1 -- nu-SVC
	% 	2 -- one-class SVM
	% 	3 -- epsilon-SVR
	% 	4 -- nu-SVR
	% -t kernel_type : set type of kernel function (default 2)
	% 	0 -- linear: u'*v
	% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
	% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
	% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
	% 	4 -- precomputed kernel (kernel values in training_set_file)
	% -d degree : set degree in kernel function (default 3)
	% -g gamma : set gamma in kernel function (default 1/num_features)
	% -r coef0 : set coef0 in kernel function (default 0)
	% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
	% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
	% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
	% -m cachesize : set cache memory size in MB (default 100)
	% -e epsilon : set tolerance of termination criterion (default 0.001)
	% -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
	% -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
	% -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
	% -v n: n-fold cross validation mode
	% -q : quiet mode (no outputs)


	% The k in the -g option means the number of attributes in the input data.

	% option -v randomly splits the data into n parts and calculates cross
	% validation accuracy/mean squared error on them.

	%TODO : test different parameters such as different kernels, different parameters for kernels such as gamma for Gaussian kernel,
	%		regularization parameter C, etc.
	%TODO : read relevant papers to see if particular kernel for person detection works and why
	%TODO : plot test/train error against different parameters to chose the best parameters
	% svmtrain(Tr.y, Tr.X, '-t 0 -v 5'); % linear kernel, 5-fold cross validation

plot_flag = 0;
predict = @(model, X) predictSVM(model, X);    
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);

learn = @(y, X) trainSVM(y, X, '-t 2 -b 1 -e 0.01');
[svmPCA.trAvgTPR, svmPCA.teAvgTPR, svmPCA.predTr, svmPCA.predTe, svmPCA.trueTr, svmPCA.trueTe] = kFoldCrossValidation(y, pcaX, 3, learn, predict, computePerformance, 'SVM RBF kernel + Exp');

learn = @(y, X) trainSVM(y, X, '-t 2 -b 1 -e 0.01');
[svmPCAexp.trAvgTPR, svmPCAexp.teAvgTPR, svmPCAexp.predTr, svmPCAexp.predTe, svmPCAexp.trueTr, svmPCAexp.trueTe] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, 'SVM RBF kernel + Exp');

methodNames = {'pca(X)', 'pca(exp(X))'};
avgTPRList = kCVevaluateMultipleMethods( cat(3, svmPCA.trueTe, svmPCAexp.trueTe), cat(3, svmPCA.predTe, svmPCAexp.predTe), true, methodNames );

    
%% kCV on different models
%TODO: check accuracy message


% Linear kernel
learn = @(y, X) trainSVM(y, X, '-t 0 -b 1 -e 0.01');
[svm1.trAvgTPR, svm1.teAvgTPR, svm1.predTr, svm1.predTe, svm1.trueTr, svm1.trueTe] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, 'SVM Linear kernel');
 
% Quadratic kernel
learn = @(y, X) trainSVM(y, X, '-t 1 -b 1 -e 0.01');
[svm2.trAvgTPR, svm2.teAvgTPR, svm2.predTr, svm2.predTe, svm2.trueTr, svm2.trueTe] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, 'SVM Polynomial kernel');
%%
% RBF kernel
plot_flag = 0;
predict = @(model, X) predictSVM(model, X);
computePerformance = @(trueOutputs, pred, plot_flag, model_name) kCVfastROC(trueOutputs, pred, plot_flag, 1, 1, model_name);

learn = @(y, X) trainSVM(y, X, '-t 2 -b 1 -e 0.01');
[svm3.trAvgTPR, svm3.teAvgTPR, svm3.predTr, svm3.predTe, svm3.trueTr, svm3.trueTe] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, 'SVM RBF kernel');
%%
methodNames = {'Linear kernel', 'Polynomial kernel', 'RBF kernel'};
avgTPRList = kCVevaluateMultipleMethods( cat(3, svm1.trueTe, svm2.trueTe, svm3.trueTe), cat(3, svm1.predTe, svm2.predTe, svm3.predTe), true, methodNames );


%% SVM RBF Kernel: Learn Gamma
    
gammaValues = [(5/(size(pcaExpX,2))),(2/(size(pcaExpX,2))), (1/size(pcaExpX,2)), (1/(2*size(pcaExpX,2))), (1/(5*size(pcaExpX,2))), (1/(10*size(pcaExpX,2)))];
[bestGamma, testTPR, trainTPR] = findGammaSVM(y, pcaExpX, 3, gammaValues);
savePlot('./report/figures/detection/svm-gamma-learningcurve.pdf','Gamma values','TPR on train (blue) and test (red)');


%% See prediction performance
fprintf('Plotting performance..\n');
% let's also see how random predicition does
randPred = rand(size(Te.y));

% and plot all together, and get the performance of each
methodNames = {'Linear Kernel', 'Polynomial Kernel', 'RBF Kernel', 'RBF + exp', 'Random'}; % this is to show it in the legend
avgTPRList = evaluateMultipleMethods( Te.y > 0, [svm1.predTe, svm2.predTe, svm3.predTe, svm4.predTe, randPred], true, methodNames ); % (true_labels, predictions, showPlot, legendNames) 
                                            

% now you can see that the performance of each method
% is in avgTPRList. You can see that random is doing very bad.
avgTPRList
