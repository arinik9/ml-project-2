function [beta, trainingErr, testErr] = penLogisticRegressionAuto(y, tX, K, lambdaValues, seed)
% Learn model parameters beta and best lambda using K-fold cross-validation
% Plot the error respective to lambda
%
% K: number of CV folds
% lambdaValues: values to try out as lambda parameters
    if(nargin < 3)
       K = 5; 
    end;
    if(nargin < 4)
       lambdaValues = logspace(-2, 2, 100); 
    end;
    if(nargin < 5)
       seed = 1; 
    end;
   
    % Step size to use for Newton's method in penalized logistic regression
    alpha = 0.5;
    
    n = length(lambdaValues);
    trainingErr = zeros(n, 1);
    testErr = zeros(n, 1);
    bestErr = -1;

    % Tryout all lambda values
    % For each value, we train with penalized logistic regression using kCV
    for i = 1:n
        lambda = lambdaValues(i);
        
        learn = @(y, tX) penLogisticRegression(y, tX, alpha, lambda);
        predict = @(tX, beta) tX * beta;
        computeError = @computeLRMse;
        
        setSeed(seed);
        [trainingErr(i), testErr(i)] = kFoldCrossValidation(y, tX, K, learn, predict, computeError);
        
        if(testErr(i) < bestErr || bestErr < 0)
            lambdaStar = lambda;
            bestErr = testErr(i, :);
        end;
        
        % Status
        %fprintf('Error for lambda = %f: %f | %f\n', lambda, trainingErr(i), testErr(i));
    end;
    
    % We have now chosen the best lambda value, we can use all the provided
    % tX as train data to learn a model
    beta = logisticRegression(y, tX, lambdaStar);
    
    % Plot evolution of train and test error with respect to lambda
    %{
    figure;
    semilogx(lambdaValues, trainingErr, '.-b');
    hold on;
    semilogx(lambdaValues, testErr, '.-r');
    xlabel('Lambda');
    ylabel('Training (blue) and test (red) error');
    %}
end




