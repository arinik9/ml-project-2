%getStartedDetection;

% Split data into train and test set given a proportion
prop = 2/3;
fprintf('Splitting into train/test with proportion %.2f..\n', prop);
[Tr.pcaX, Tr.y, Te.pcaX, Te.y] = splitDataDetection(y, pcaX, prop);

fprintf('Splitting into train/test with proportion %.2f..\n', prop);
[Tr.pcaExpX, Tr.expy, Te.pcaExpX, Te.expy] = splitDataDetection(y, pcaExpX, prop);

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

%% SVM RBF Kernel: Learn Gamma
    
gammaValues = [(5/(size(pcaX,2))),(2/(size(pcaX,2))), (1/size(pcaX,2)), (1/(2*size(pcaX,2))), (1/(5*size(pcaX,2))), (1/(10*size(pcaX,2)))];
[bestGamma, testTPR, trainTPR] = findGammaSVM(y, pcaX, 3, gammaValues);
    

%% kCV on different models
%TODO: check accuracy message

plot_flag = 1;

learn = @(y, X) trainSVM(y, X, '-t 0 -b 1 -e 0.01');
predict = @(model, X) predictSVM(model, X);
computePerformance = @(trueOutputs, pred, model_name) kCVfastROC(trueOutputs, pred, model_name, 1);
[svm1.trAvgTPR, svm1.teAvgTPR, svm1.predTr, svm1.predTe] = kFoldCrossValidation(y, pcaX, 3, learn, predict, computePerformance, 'SVM Linear kernel');

learn = @(y, X) trainSVM(y, X, '-t 1 -b 1 -e 0.01');
predict = @(model, X) predictSVM(model, X);
computePerformance = @(trueOutputs, pred, model_name) kCVfastROC(trueOutputs, pred, model_name, 1);
[svm2.trAvgTPR, svm2.teAvgTPR, svm2.predTr, svm2.predTe] = kFoldCrossValidation(y, pcaX, 3, learn, predict, computePerformance, 'SVM Polynomial kernel');

learn = @(y, X) trainSVM(y, X, '-t 2 -b 1 -e 0.01');
predict = @(model, X) predictSVM(model, X);
computePerformance = @(trueOutputs, pred, model_name) kCVfastROC(trueOutputs, pred, model_name, 1);
[svm3.trAvgTPR, svm3.teAvgTPR, svm3.predTr, svm3.predTe] = kFoldCrossValidation(y, pcaX, 3, learn, predict, computePerformance, 'SVM RBF kernel');

learn = @(y, X) trainSVM(y, X, '-t 2 -b 1 -e 0.01');
predict = @(model, X) predictSVM(model, X);
computePerformance = @(trueOutputs, pred, model_name) kCVfastROC(trueOutputs, pred, model_name, 1);
[svm4.trAvgTPR, svm4.teAvgTPR, svm4.predTr, svm4.predTe] = kFoldCrossValidation(y, pcaExpX, 3, learn, predict, computePerformance, 'SVM RBF kernel + Exp');

%% Try different models

fprintf('train SVM1...\n')
svm1 = trainSVM(Tr.y, Tr.pcaX, '-t 0 -b 1 -e 0.01'); % C-SVC linear kernel, with probabilities, etc
fprintf('predict SVM1...\n')
svmPred1 = predictSVM(svm1, Te.pcaX, Te.y);

fprintf('train SVM2...\n')
svm2 = trainSVM(Tr.y, Tr.pcaX, '-t 1 -b 1 -e 0.01'); % C-SVC quadratic kernel, with probabilities, etc
fprintf('predict SVM2...\n')
svmPred2 = predictSVM(svm2, Te.pcaX, Te.y);

fprintf('train SVM3...\n')
svm3 = trainSVM(Tr.y, Tr.pcaX, '-t 2 -b 1 -e 0.01'); % C-SVC RFB kernel, with probabilities, etc
fprintf('predict SVM3...\n')
svmPred3 = predictSVM(svm3, Te.pcaX, Te.y);

fprintf('train SVMExp...\n')
svmExp = trainSVM(Tr.expy, Tr.pcaExpX, '-t 2 -b 1 -e 0.01'); % C-SVC RFB kernel exp tranfo, with probabilities, etc
fprintf('predict SVMExp...\n')
svmPredExp = predictSVM(svmExp, Te.pcaExpX, Te.expy);


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
