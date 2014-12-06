clearvars;

addpath(genpath('../../data'));
addpath(genpath('../../toolbox/Piotr'));
addpath(genpath('../../toolbox/DeepLearnToolbox'));
addpath(genpath('../../toolbox/libsvm'));



load train_feats;

%% -- Generate feature vectors (so each one is a row of X)
fprintf('Generating feature vectors..\n');
D = numel(feats{1});  % feature dimensionality
X = zeros([length(feats) D]);

for i=1:length(feats)
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end



%% Randomly split data into train/test sets according to a fixed proportion, set aside the test data
fprintf('Splitting into train/test..\n');
[Tr.X, Tr.y, Te.X, Te.y] = splitDataDetection(labels, X, 0.7);
	


% normalize data
[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

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



% create a model with the tuned parameters
model = svmtrain(Tr.y, Tr.normX, '-t 0 -b 1 -e 0.01'); % linear kernel, with probabilities, etc

% predict on test data
[predict_label, accuracy, prob_estimates] = svmpredict(Te.y, Te.normX, model, '-b 1');

% get scores(not labels) for ROC
svmPredict  = prob_estimates(:, 1) - prob_estimates(:, 2);

%% See prediction performance
fprintf('Plotting performance..\n');
% let's also see how random predicition does
randPred = rand(size(Te.y));

% and plot all together, and get the performance of each
methodNames = {'SVM', 'Random'}; % this is to show it in the legend
avgTPRList = evaluateMultipleMethods( Te.y > 0, [svmPredict, randPred], true, methodNames ); % (true_labels, predictions, showPlot, legendNames) 
                                            

% now you can see that the performance of each method
% is in avgTPRList. You can see that random is doing very bad.
avgTPRList
